# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import sys
import types
from typing import Any

import pytest


def _load_tp_ipc_util(monkeypatch):
    fake_torch: Any = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        synchronize=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_vmm_ops: Any = types.ModuleType("kvcached.vmm_ops")
    fake_vmm_ops.kv_tensors_created = lambda *args, **kwargs: True
    fake_vmm_ops.map_to_kv_tensors = lambda *args, **kwargs: True
    fake_vmm_ops.map_to_kv_tensors_with_result = lambda offsets, group_id=0: (True, list(offsets))
    fake_vmm_ops.unmap_from_kv_tensors = lambda *args, **kwargs: True
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", fake_vmm_ops)

    module = importlib.import_module("kvcached.tp_ipc_util")
    return importlib.reload(module)


def test_partial_pp_map_rolls_back_only_new_offsets(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    calls = []

    async def fake_send(rank, message, pp_rank=0):
        calls.append((pp_rank, rank, message["cmd"], tuple(message["offsets"])))
        if message["cmd"] == "unmap_from_kv_tensors":
            return {"status": "success"}
        if (pp_rank, rank) == (0, 1):
            return {"status": "error", "message": "map failed"}
        newly_mapped = {
            (0, 0): [0],
            (1, 0): [0, 2],
            (1, 1): [],
        }[(pp_rank, rank)]
        return {
            "status": "success",
            "newly_mapped_offsets": newly_mapped,
        }

    monkeypatch.setenv("KVCACHED_PP_SIZE", "2")
    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)

    with pytest.raises(RuntimeError, match="pp0/rank1"):
        asyncio.run(
            module._broadcast_map_to_kv_tensors(
                tp_size=2,
                offsets=[0, 2],
                pp_rank=-1,
                group_id=7,
            )
        )

    rollback_calls = [call for call in calls if call[2] == "unmap_from_kv_tensors"]
    assert rollback_calls == [
        (0, 0, "unmap_from_kv_tensors", (0,)),
        (1, 0, "unmap_from_kv_tensors", (0, 2)),
    ]


def test_partial_map_reports_rollback_failure(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)

    async def fake_send(rank, message, pp_rank=0):
        if message["cmd"] == "unmap_from_kv_tensors":
            return {"status": "error", "message": "rollback failed"}
        if rank == 1:
            return {"status": "error", "message": "map failed"}
        return {"status": "success", "newly_mapped_offsets": [0]}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)

    with pytest.raises(RuntimeError, match="rollback failures.*rollback failed"):
        asyncio.run(module._broadcast_map_to_kv_tensors(2, [0]))


def test_lost_map_response_is_reported_as_unknown_state(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)

    async def fake_send(rank, message, pp_rank=0):
        if message["cmd"] == "map_to_kv_tensors" and rank == 1:
            raise ConnectionError("response lost")
        return {"status": "success", "newly_mapped_offsets": [0]}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)

    with pytest.raises(RuntimeError, match="state_consistency_unknown.*pp0/rank1"):
        asyncio.run(module._broadcast_map_to_kv_tensors(2, [0]))


def test_successful_map_does_not_issue_rollback(monkeypatch):
    module = _load_tp_ipc_util(monkeypatch)
    commands = []

    async def fake_send(rank, message, pp_rank=0):
        commands.append(message["cmd"])
        return {"status": "success", "newly_mapped_offsets": [0]}

    monkeypatch.setattr(module, "_send_and_receive_message", fake_send)
    asyncio.run(module._broadcast_map_to_kv_tensors(2, [0]))

    assert commands == ["map_to_kv_tensors", "map_to_kv_tensors"]
