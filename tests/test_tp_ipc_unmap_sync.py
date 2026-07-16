# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

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
    fake_vmm_ops.unmap_from_kv_tensors = lambda *args, **kwargs: True
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", fake_vmm_ops)

    mod = importlib.import_module("kvcached.tp_ipc_util")
    return importlib.reload(mod)


def test_tp_unmap_syncs_cuda_by_default(monkeypatch):
    mod = _load_tp_ipc_util(monkeypatch)
    calls = []

    fake_torch: Any = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("sync"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    mod._sync_before_unmap()

    assert calls == ["sync"]


def test_tp_map_syncs_cuda_by_default(monkeypatch):
    mod = _load_tp_ipc_util(monkeypatch)
    calls = []

    fake_torch: Any = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("sync"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    mod._sync_after_map()

    assert calls == ["sync"]


def test_tp_unmap_sync_can_be_disabled(monkeypatch):
    mod = _load_tp_ipc_util(monkeypatch)
    calls = []

    fake_torch: Any = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("sync"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("KVCACHED_SYNC_BEFORE_TP_UNMAP", "false")

    mod._sync_before_unmap()

    assert calls == []


def test_tp_map_sync_can_be_disabled(monkeypatch):
    mod = _load_tp_ipc_util(monkeypatch)
    calls = []

    fake_torch: Any = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: calls.append("sync"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("KVCACHED_SYNC_AFTER_TP_MAP", "false")

    mod._sync_after_map()

    assert calls == []


def test_tp_map_sync_propagates_cuda_errors(monkeypatch):
    mod = _load_tp_ipc_util(monkeypatch)

    fake_torch: Any = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: (_ for _ in ()).throw(RuntimeError("cuda failed")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="cuda failed"):
        mod._sync_after_map()
