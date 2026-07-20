# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import sys
from types import ModuleType


def _load_tp_ipc_util(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    vmm_ops = ModuleType("kvcached.vmm_ops")
    setattr(vmm_ops, "kv_tensors_created", lambda group_id=0: True)
    setattr(vmm_ops, "map_to_kv_tensors", lambda offsets, group_id=0: True)
    setattr(vmm_ops, "unmap_from_kv_tensors", lambda offsets, group_id=0: True)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops)
    sys.modules.pop("kvcached.tp_ipc_util", None)
    return importlib.import_module("kvcached.tp_ipc_util")


def test_broadcast_map_rolls_back_every_rank_after_partial_failure(monkeypatch):
    tp_ipc = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def fake_send(rank, message, pp_rank=0):
        calls.append((rank, message["cmd"], tuple(message["offsets"]), pp_rank))
        if message["cmd"] == "map_to_kv_tensors" and rank == 1:
            return {"status": "error", "message": "capacity_exhausted"}
        if message["cmd"] == "unmap_from_kv_tensors" and rank == 2:
            return {"status": "error", "message": "worker unavailable"}
        return {"status": "success"}

    monkeypatch.setattr(tp_ipc, "_send_and_receive_message", fake_send)

    try:
        asyncio.run(
            tp_ipc._broadcast_map_to_kv_tensors(
                tp_size=3, offsets=[0, 2 * 1024 * 1024], pp_rank=2, group_id=7
            )
        )
    except RuntimeError as exc:
        assert "worker 1" in str(exc)
        assert "capacity_exhausted" in str(exc)
        assert "rollback failures" in str(exc)
        assert "worker unavailable" in str(exc)
    else:
        raise AssertionError("partial TP map failure did not propagate")

    rollback_ranks = {
        rank for rank, command, _offsets, _pp_rank in calls
        if command == "unmap_from_kv_tensors"
    }
    assert rollback_ranks == {0, 1, 2}


def test_broadcast_map_does_not_rollback_success(monkeypatch):
    tp_ipc = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def fake_send(rank, message, pp_rank=0):
        calls.append((rank, message["cmd"]))
        return {"status": "success"}

    monkeypatch.setattr(tp_ipc, "_send_and_receive_message", fake_send)
    asyncio.run(tp_ipc._broadcast_map_to_kv_tensors(2, [0]))

    assert calls == [(0, "map_to_kv_tensors"), (1, "map_to_kv_tensors")]
