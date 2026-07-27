# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import importlib
import socket
import sys
import time
import types

import pytest

sys.modules.setdefault("torch", types.ModuleType("torch"))
vmm_ops = types.ModuleType("kvcached.vmm_ops")
setattr(vmm_ops, "kv_tensors_created", lambda *args, **kwargs: False)
setattr(vmm_ops, "map_to_kv_tensors", lambda *args, **kwargs: True)
setattr(vmm_ops, "unmap_from_kv_tensors", lambda *args, **kwargs: True)
sys.modules.setdefault("kvcached.vmm_ops", vmm_ops)

tp_ipc_util = importlib.import_module("kvcached.tp_ipc_util")


@pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="requires Unix sockets")
def test_worker_listener_reports_cuda_memory(monkeypatch, tmp_path):
    monkeypatch.setattr(tp_ipc_util, "SOCKET_DIR", str(tmp_path))

    torch = types.ModuleType("torch")
    setattr(
        torch,
        "cuda",
        types.SimpleNamespace(
            mem_get_info=lambda: (1234, 5678),
            current_device=lambda: 2,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    tp_ipc_util.start_worker_listener_thread(rank=0, pp_rank=0)
    socket_path = tmp_path / "w0.sock"
    deadline = time.monotonic() + 2.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert tp_ipc_util.query_worker_cuda_mem_get_info(
        rank=0,
        pp_rank=0,
        timeout=1.0,
    ) == (1234, 5678)
