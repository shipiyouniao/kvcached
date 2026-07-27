# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import pickle
import socket
import threading
import uuid
from typing import Any, Dict, cast

from kvcached import vmm_ops
from kvcached.utils import DEFAULT_IPC_NAME

kv_tensors_created = vmm_ops.kv_tensors_created
map_to_kv_tensors = vmm_ops.map_to_kv_tensors
unmap_from_kv_tensors = vmm_ops.unmap_from_kv_tensors


def _map_to_kv_tensors_with_result(offsets: list[int], group_id: int) -> tuple[bool, list[int]]:
    operation = getattr(vmm_ops, "map_to_kv_tensors_with_result", None)
    if operation is None:
        return bool(map_to_kv_tensors(offsets, group_id=group_id)), list(offsets)
    success, newly_mapped = operation(offsets, group_id=group_id)
    return bool(success), [int(offset) for offset in newly_mapped]


def _get_socket_dir_name() -> str:
    """
    Build a human-readable, IPC-name-based directory with a short hash suffix.

    This keeps the original text-based IPC name visible while adding a hash
    for extra uniqueness. The hash is deterministic so all workers in the same
    engine instance agree on the directory.
    """
    # Deterministic short hash derived from the base name
    suffix = uuid.uuid5(uuid.NAMESPACE_DNS, DEFAULT_IPC_NAME).hex[:8]
    return f"kvcached-tp-{DEFAULT_IPC_NAME}-{suffix}"


# Socket directory for tensor parallel (TP) worker communication.
# Unix domain socket paths are limited to 108 characters on Linux, so we keep
# the directory name short and validate the final socket path length below.
SOCKET_DIR = os.path.join("/tmp", _get_socket_dir_name())


def _target_pp_ranks(pp_rank: int) -> range:
    if pp_rank >= 0:
        return range(pp_rank, pp_rank + 1)

    pp_size = int(os.getenv("KVCACHED_PP_SIZE", "1"))
    if pp_size <= 0:
        raise ValueError(f"KVCACHED_PP_SIZE must be positive, got {pp_size}")
    return range(pp_size)


def _env_falsey(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"0", "false", "no", "off"}


def _sync_before_unmap() -> None:
    """Match PageAllocator's single-process async unmap safety in TP mode."""
    if _env_falsey("KVCACHED_SYNC_BEFORE_TP_UNMAP"):
        return

    _sync_cuda(suppress_errors=True)


def _sync_after_map() -> None:
    """Make newly mapped TP pages visible before schedulers touch them."""
    if _env_falsey("KVCACHED_SYNC_AFTER_TP_MAP"):
        return

    _sync_cuda(suppress_errors=False)


def _sync_cuda(*, suppress_errors: bool) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        if not suppress_errors:
            raise
        # Unmap will report the real failure if the CUDA context is already
        # unhealthy.  Do not hide the original unmap response behind sync
        # bookkeeping.
        pass


def get_worker_socket_path(rank: int, pp_rank: int = 0) -> str:
    """
    Get the path for the worker socket, namespaced by pp_rank.
    Each PP stage uses its own subdirectory to avoid EADDRINUSE races
    when multiple stages start simultaneously (SGLang PP behaviour).

    The full path is guaranteed to be <= 108 characters (Unix domain socket limit).
    """
    if pp_rank > 0:
        socket_path = os.path.join(SOCKET_DIR, f"pp{pp_rank}", f"w{rank}.sock")
    else:
        socket_path = os.path.join(SOCKET_DIR, f"w{rank}.sock")

    if len(socket_path) > 108:
        raise RuntimeError(
            f"Socket path too long ({len(socket_path)} chars, max 108): {socket_path}"
        )

    return socket_path


# NOTE: All messages exchanged through the IPC layer are dictionaries with
# string keys and arbitrary JSON-serialisable (picklable) values.
Message = Dict[str, Any]


def send_msg(sock: socket.socket, msg: Message) -> None:
    """
    Send a message through the socket.
    The message is serialized using pickle.
    """
    data = pickle.dumps(msg)
    sock.sendall(len(data).to_bytes(4, 'big') + data)


# The receive side mirrors *send_msg* and therefore also returns a *Message*.
def recv_msg(sock: socket.socket) -> Message:
    """
    Receive a message from the socket.
    The message is deserialized using pickle.
    """
    length_bytes = sock.recv(4)
    if not length_bytes:
        raise ConnectionError("Socket connection closed")
    if not len(length_bytes) == 4:
        raise ValueError("Received incomplete length bytes from socket")
    length = int.from_bytes(length_bytes, 'big')
    if length <= 0:
        raise ValueError("Received invalid length for message")
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError(
                "Socket connection closed while receiving data")
        data += chunk
    if len(data) != length:
        raise ValueError("Received data length does not match expected length")
    return cast(Message, pickle.loads(data))


def start_worker_listener_thread(rank: int, pp_rank: int = 0):
    """
    Start a thread that listens for messages on the worker socket.
    pp_rank is used to create a PP-stage-specific subdirectory so that
    concurrent SGLang PP stages do not bind the same socket path.
    """
    socket_dir = os.path.join(SOCKET_DIR, f"pp{pp_rank}") if pp_rank > 0 else SOCKET_DIR
    os.makedirs(socket_dir, exist_ok=True)
    socket_path = get_worker_socket_path(rank, pp_rank)

    if os.path.exists(socket_path):
        try:
            os.remove(socket_path)
        except OSError as e:
            print(f"Error removing existing socket file {socket_path}: {e}")

    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(socket_path)
    server_sock.listen()

    def listen_loop():
        print(f"Worker {rank} IPC listener started at {socket_path}")
        while True:
            conn, _ = server_sock.accept()
            try:
                msg: Message = recv_msg(conn)
                # print(f"Worker {rank} received message: {msg}")
                group_id: int = msg.get("group_id", 0)
                if msg["cmd"] == "map_to_kv_tensors":
                    success, newly_mapped = _map_to_kv_tensors_with_result(msg["offsets"], group_id)
                    if not success:
                        raise RuntimeError(f"Failed to map KV tensors for group_id={group_id}")
                    try:
                        _sync_after_map()
                    except Exception as map_error:
                        try:
                            if newly_mapped:
                                _sync_before_unmap()
                                if not unmap_from_kv_tensors(newly_mapped, group_id=group_id):
                                    raise RuntimeError("map rollback unmap returned false")
                        except Exception as rollback_error:
                            raise RuntimeError(
                                "state_inconsistency: map synchronization failed: "
                                f"{map_error}; rollback failed: {rollback_error}"
                            ) from map_error
                        raise
                    send_msg(
                        conn,
                        {
                            "status": "success",
                            "newly_mapped_offsets": newly_mapped,
                        },
                    )
                elif msg["cmd"] == "unmap_from_kv_tensors":
                    _sync_before_unmap()
                    if not unmap_from_kv_tensors(msg["offsets"], group_id=group_id):
                        raise RuntimeError(
                            f"Failed to unmap KV tensors for group_id={group_id}"
                        )
                    send_msg(conn, {"status": "success"})
                elif msg["cmd"] == "kv_tensors_created":
                    created: bool = kv_tensors_created(group_id=group_id)
                    send_msg(conn, {"status": "success", "created": created})
                else:
                    send_msg(conn, {
                        "status": "error",
                        "message": "Unknown command"
                    })
            except Exception as e:
                print(f"Worker {rank} error processing message: {e}")
                send_msg(conn, {"status": "error", "message": str(e)})
            finally:
                conn.close()

    t = threading.Thread(target=listen_loop, daemon=True)
    t.start()


async def _send_and_receive_message(rank: int, message: Message, pp_rank: int = 0) -> Message:
    """
    Send a message to the worker and receive a response asynchronously.
    """
    socket_path = get_worker_socket_path(rank, pp_rank)
    reader, writer = await asyncio.open_unix_connection(socket_path)

    try:
        # Send map command
        data = pickle.dumps(message)
        writer.write(len(data).to_bytes(4, 'big') + data)
        await writer.drain()

        # Read the length of the response from worker
        length_bytes = await reader.readexactly(4)
        length = int.from_bytes(length_bytes, 'big')

        # Read the actual response data
        data = await reader.readexactly(length)
        return cast(Message, pickle.loads(data))
    finally:
        writer.close()
        await writer.wait_closed()


async def _broadcast_map_to_kv_tensors(tp_size: int,
                                       offsets: list[int],
                                       pp_rank: int = 0,
                                       group_id: int = 0) -> None:
    """
    Broadcast the "map_to_kv_tensors" operation to all workers concurrently.
    """
    map_message = {"cmd": "map_to_kv_tensors", "offsets": offsets,
                   "group_id": group_id}
    targets = [
        (target_pp_rank, rank)
        for target_pp_rank in _target_pp_ranks(pp_rank)
        for rank in range(tp_size)
    ]
    tasks = [
        _send_and_receive_message(rank, map_message, target_pp_rank)
        for target_pp_rank, rank in targets
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    failures: list[str] = []
    unknown_targets: list[str] = []
    rollback_targets: list[tuple[int, int, list[int]]] = []
    for (target_pp_rank, rank), response in zip(targets, responses):
        if isinstance(response, Exception):
            target = f"pp{target_pp_rank}/rank{rank}"
            failures.append(f"Worker {target} failed to map: {response}")
            unknown_targets.append(target)
        elif not isinstance(response, dict) or response.get("status") != "success":
            failures.append(f"Worker pp{target_pp_rank}/rank{rank} failed to map: {response}")
        else:
            newly_mapped = response.get("newly_mapped_offsets")
            if newly_mapped is None:
                newly_mapped = []
            rollback_targets.append(
                (
                    target_pp_rank,
                    rank,
                    [int(offset) for offset in newly_mapped],
                )
            )

    if not failures:
        return

    rollback_tasks = []
    rollback_task_targets = []
    for target_pp_rank, rank, newly_mapped in rollback_targets:
        if not newly_mapped:
            continue
        rollback_task_targets.append((target_pp_rank, rank))
        rollback_tasks.append(
            _send_and_receive_message(
                rank,
                {
                    "cmd": "unmap_from_kv_tensors",
                    "offsets": newly_mapped,
                    "group_id": group_id,
                },
                target_pp_rank,
            )
        )

    rollback_failures: list[str] = []
    if rollback_tasks:
        rollback_responses = await asyncio.gather(*rollback_tasks, return_exceptions=True)
        for (target_pp_rank, rank), response in zip(rollback_task_targets, rollback_responses):
            if isinstance(response, Exception):
                rollback_failures.append(f"pp{target_pp_rank}/rank{rank}: {response}")
            elif not isinstance(response, dict) or response.get("status") != "success":
                rollback_failures.append(f"pp{target_pp_rank}/rank{rank}: {response}")

    message = "; ".join(failures)
    if unknown_targets:
        message += "; state_consistency_unknown for workers with lost responses: " + ", ".join(
            unknown_targets
        )
    if rollback_failures:
        message += "; rollback failures: " + "; ".join(rollback_failures)
    raise RuntimeError(message)


async def _broadcast_unmap_from_kv_tensors(tp_size: int,
                                           offsets: list[int],
                                           pp_rank: int = 0,
                                           group_id: int = 0) -> None:
    """
    Broadcast the "unmap_from_kv_tensors" operation to all workers concurrently.
    """
    unmap_message = {"cmd": "unmap_from_kv_tensors", "offsets": offsets,
                     "group_id": group_id}
    targets = [
        (target_pp_rank, rank)
        for target_pp_rank in _target_pp_ranks(pp_rank)
        for rank in range(tp_size)
    ]
    tasks = [
        _send_and_receive_message(rank, unmap_message, target_pp_rank)
        for target_pp_rank, rank in targets
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    for (target_pp_rank, rank), response in zip(targets, responses):
        if isinstance(response, Exception):
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to unmap: {response}"
            )
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to unmap: {response}"
            )


async def _broadcast_kv_tensors_created(tp_size: int,
                                        pp_rank: int = 0,
                                        group_id: int = 0) -> bool:
    """
    Broadcast the "kv_tensors_created" operation to all workers concurrently.
    Returns True if all workers report that KV tensors are created, False otherwise.
    """
    check_message = {"cmd": "kv_tensors_created", "group_id": group_id}
    targets = [
        (target_pp_rank, rank)
        for target_pp_rank in _target_pp_ranks(pp_rank)
        for rank in range(tp_size)
    ]
    tasks = [
        _send_and_receive_message(rank, check_message, target_pp_rank)
        for target_pp_rank, rank in targets
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)
    all_created = True
    for (target_pp_rank, rank), response in zip(targets, responses):
        if isinstance(response, Exception):
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to check "
                f"KV tensors created: {response}"
            )
        elif not isinstance(response,
                            dict) or response.get("status") != "success":
            raise RuntimeError(
                f"Worker pp{target_pp_rank}/rank{rank} failed to check "
                f"KV tensors created: {response}"
            )
        elif not response.get("created", False):
            all_created = False

    return all_created


# Wrapper functions to call the async function from sync code
def broadcast_map_to_kv_tensors(tp_size: int, offsets: list[int],
                                pp_rank: int = 0,
                                group_id: int = 0) -> None:
    asyncio.run(_broadcast_map_to_kv_tensors(tp_size, offsets, pp_rank,
                                             group_id))


def broadcast_unmap_from_kv_tensors(tp_size: int, offsets: list[int],
                                    pp_rank: int = 0,
                                    group_id: int = 0) -> None:
    asyncio.run(_broadcast_unmap_from_kv_tensors(tp_size, offsets, pp_rank,
                                                 group_id))


def broadcast_kv_tensors_created(tp_size: int, pp_rank: int = 0,
                                 group_id: int = 0) -> bool:
    return asyncio.run(_broadcast_kv_tensors_created(tp_size, pp_rank,
                                                     group_id))
