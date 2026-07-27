# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import threading
import types
from typing import Any

import pytest


def _import_kv_cache_manager(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))

    vmm_ops: Any = types.ModuleType("kvcached.vmm_ops")
    vmm_ops.kv_tensors_created = lambda *args, **kwargs: False
    vmm_ops.map_to_kv_tensors = lambda *args, **kwargs: True
    vmm_ops.unmap_from_kv_tensors = lambda *args, **kwargs: True
    vmm_ops.PageAllocator = object
    vmm_ops.InternalPage = object
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", vmm_ops)

    interfaces: Any = types.ModuleType("kvcached.integration.vllm.interfaces")
    interfaces.should_use_worker_ipc = lambda: False
    monkeypatch.setitem(sys.modules, "kvcached.integration.vllm.interfaces", interfaces)

    from kvcached import kv_cache_manager

    return kv_cache_manager


def test_post_init_timeout_keeps_last_observed_error(monkeypatch):
    kv_cache_manager = _import_kv_cache_manager(monkeypatch)
    manager = kv_cache_manager.KVCacheManager.__new__(kv_cache_manager.KVCacheManager)
    manager.null_block = None
    manager.world_size = 1
    manager.pp_rank = 0
    manager.group_id = 0
    manager._post_init_done = threading.Event()

    calls = 0

    def fake_kv_tensors_created(group_id=0):
        nonlocal calls
        assert group_id == 0
        calls += 1
        if calls == 1:
            raise RuntimeError("kv tensor map failed")
        return False

    monkeypatch.setattr(kv_cache_manager, "kv_tensors_created", fake_kv_tensors_created)
    monkeypatch.setattr(kv_cache_manager, "KV_TENSOR_WAIT_TIMEOUT", 0.002)
    monkeypatch.setattr(kv_cache_manager.time, "sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="last error: kv tensor map failed"):
        manager._post_init()

    assert manager._post_init_done.is_set()


def test_reserve_null_block_waits_for_transient_capacity(monkeypatch):
    kv_cache_manager = _import_kv_cache_manager(monkeypatch)
    manager = kv_cache_manager.KVCacheManager.__new__(kv_cache_manager.KVCacheManager)
    manager.reserve_null_block = True
    manager.null_block = None

    available = iter((0, 0, 1))
    alloc_calls = []
    manager.available_size = lambda: next(available)

    def fake_alloc(need_size, _skip_wait=False):
        alloc_calls.append((need_size, _skip_wait))
        return [0]

    manager._alloc = fake_alloc
    monkeypatch.setattr(kv_cache_manager.time, "sleep", lambda _: None)

    manager._reserve_null_block()

    assert manager.null_block == [0]
    assert alloc_calls == [(1, True)]


def test_broadcast_callbacks_preserve_runtime_group_context(monkeypatch):
    kv_cache_manager = _import_kv_cache_manager(monkeypatch)
    calls = []

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(kv_cache_manager.threading, "Thread", FakeThread)

    class FakePageAllocator:
        def __init__(self, *args, **kwargs):
            self.map_callback = None
            self.unmap_callback = None

        def set_should_use_worker_ipc_callback(self, callback):
            self.should_use_worker_ipc_callback = callback

        def set_broadcast_map_callback(self, callback):
            self.map_callback = callback

        def set_broadcast_unmap_callback(self, callback):
            self.unmap_callback = callback

        def start_prealloc_thread(self):
            pass

    monkeypatch.setattr(kv_cache_manager, "PageAllocator", FakePageAllocator)
    monkeypatch.setattr(
        kv_cache_manager, "broadcast_kv_tensors_created", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(kv_cache_manager, "KV_TENSOR_WAIT_TIMEOUT", 0.001)
    monkeypatch.setattr(kv_cache_manager.time, "sleep", lambda _: None)

    import kvcached.tp_ipc_util as tp_ipc_util

    monkeypatch.setattr(
        tp_ipc_util,
        "broadcast_map_to_kv_tensors",
        lambda world_size, offsets, pp_rank=0, group_id=0: calls.append(
            ("map", world_size, offsets, pp_rank, group_id)
        ),
    )
    monkeypatch.setattr(
        tp_ipc_util,
        "broadcast_unmap_from_kv_tensors",
        lambda world_size, offsets, pp_rank=0, group_id=0: calls.append(
            ("unmap", world_size, offsets, pp_rank, group_id)
        ),
    )

    manager = kv_cache_manager.KVCacheManager(
        num_blocks=4,
        block_size=1,
        cell_size=1,
        num_layers=1,
        world_size=4,
        pp_rank=2,
        async_sched=True,
        group_id=1007,
    )

    manager.page_allocator.map_callback(4, [1, 2, 3])
    manager.page_allocator.unmap_callback(4, [4, 5])

    assert calls == [
        ("map", 4, [1, 2, 3], 2, 1007),
        ("unmap", 4, [4, 5], 2, 1007),
    ]


def test_enginecore_publishes_worker_meminfo_snapshot_from_python(monkeypatch):
    kv_cache_manager = _import_kv_cache_manager(monkeypatch)
    observed: dict[str, Any] = {}

    class FakePageAllocator:
        def __init__(self, *args, **kwargs):
            observed["constructor"] = kwargs

        def update_mem_info_snapshot(self, avail_bytes, total_bytes):
            observed["mem_info_snapshot"] = (avail_bytes, total_bytes)

        def set_should_use_worker_ipc_callback(self, callback):
            observed["worker_ipc_callback"] = callback

        def set_broadcast_map_callback(self, callback):
            pass

        def set_broadcast_unmap_callback(self, callback):
            pass

    monkeypatch.setattr(kv_cache_manager, "PageAllocator", FakePageAllocator)
    monkeypatch.setattr(kv_cache_manager, "ENGINECORE_NO_CUDA", True)
    monkeypatch.setattr(kv_cache_manager, "MEMINFO_PROVIDER", "worker")

    class FakeThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(kv_cache_manager.threading, "Thread", FakeThread)

    import kvcached.meminfo_provider as meminfo_provider

    def query_worker(world_size, pp_rank):
        observed["query"] = (world_size, pp_rank)
        return 123, 456

    monkeypatch.setattr(meminfo_provider, "query_mem_info", query_worker)

    manager = kv_cache_manager.KVCacheManager(
        num_blocks=4,
        block_size=1,
        cell_size=1,
        num_layers=1,
        pp_rank=-1,
    )

    assert observed["constructor"]["cuda_control_plane"] is False
    manager._refresh_mem_info_snapshot()
    assert observed["query"] == (1, -1)
    assert observed["mem_info_snapshot"] == (123, 456)


def test_enginecore_rejects_local_meminfo_without_cuda(monkeypatch):
    kv_cache_manager = _import_kv_cache_manager(monkeypatch)

    class FakePageAllocator:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(kv_cache_manager, "PageAllocator", FakePageAllocator)
    monkeypatch.setattr(kv_cache_manager, "ENGINECORE_NO_CUDA", True)
    monkeypatch.setattr(kv_cache_manager, "MEMINFO_PROVIDER", "local")

    with pytest.raises(
        kv_cache_manager.KVCachedConfigError,
        match="requires KVCACHED_MEMINFO_PROVIDER=worker",
    ):
        kv_cache_manager.KVCacheManager(
            num_blocks=4,
            block_size=1,
            cell_size=1,
            num_layers=1,
        )
