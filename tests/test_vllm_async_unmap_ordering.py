# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from collections import deque
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from unittest import mock

import pytest


def _load_patches(monkeypatch):
    torch = mock.MagicMock()
    torch.__version__ = "2.6.0"
    torch.__spec__ = ModuleSpec("torch", loader=None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "kvcached.vmm_ops", mock.MagicMock())

    from kvcached.integration.vllm import patches

    return patches


class FakeManager:
    def __init__(self):
        self.marker = 0
        self.released = []

    def retire(self):
        self.marker += 1

    def capture_physical_release_marker(self):
        return self.marker

    def release_retired_pages_through(self, marker):
        self.released.append(marker)


def _patch_engine(monkeypatch, original_step):
    patches = _load_patches(monkeypatch)
    engine_mod = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        step_with_batch_queue = original_step

    setattr(engine_mod, "EngineCore", EngineCore)
    assert patches.EngineCorePatch().patch_async_batch_lifetime(engine_mod)
    return EngineCore


def _engine(EngineCore, manager, queue):
    engine = EngineCore()
    engine.batch_queue = queue
    engine.scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            block_pool=SimpleNamespace(kv_cache_manager=manager)
        )
    )
    return engine


def test_completed_batch_releases_only_pages_retired_before_call(monkeypatch):
    manager = FakeManager()
    manager.retire()

    def original_step(self):
        manager.retire()
        return ({}, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, deque([object()]))

    engine.step_with_batch_queue()

    assert manager.released == [1]


def test_final_completed_batch_releases_all_retired_pages(monkeypatch):
    manager = FakeManager()
    manager.retire()

    def original_step(self):
        manager.retire()
        self.batch_queue.clear()
        return ({}, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, deque([object()]))

    engine.step_with_batch_queue()

    assert manager.released == [2]


def test_queue_submission_without_completion_does_not_release_pages(monkeypatch):
    manager = FakeManager()

    def original_step(self):
        manager.retire()
        self.batch_queue.appendleft(object())
        return (None, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, deque())

    engine.step_with_batch_queue()

    assert manager.released == []


def test_retired_pages_wait_for_every_older_inflight_batch(monkeypatch):
    manager = FakeManager()

    def original_step(self):
        self.batch_queue.pop()
        return ({}, True)

    EngineCore = _patch_engine(monkeypatch, original_step)
    engine = _engine(EngineCore, manager, deque([object(), object(), object()]))
    manager.retire()

    engine.step_with_batch_queue()
    engine.step_with_batch_queue()
    assert manager.released == []

    engine.step_with_batch_queue()
    assert manager.released == [1]


def test_engine_ordered_unmap_publishes_capacity_change(monkeypatch):
    patches = _load_patches(monkeypatch)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)

    interfaces = __import__(
        "kvcached.integration.vllm.interfaces", fromlist=["init_kvcached"]
    )
    monkeypatch.setattr(interfaces, "init_kvcached", mock.Mock())
    tp_ipc_util = __import__("kvcached.tp_ipc_util", fromlist=["unused"])
    notify = mock.Mock(return_value=True)
    monkeypatch.setattr(
        tp_ipc_util, "notify_physical_growth_capacity_changed", notify
    )

    class PageAllocator:
        callback = None

        def set_broadcast_unmap_callback(self, callback):
            self.callback = callback

    manager = SimpleNamespace(
        group_id=17,
        pp_rank=-1,
        page_allocator=PageAllocator(),
        _increment_operation_counter=mock.Mock(),
    )

    class Executor:
        def collective_rpc(self, method, *, args):
            assert method is patches._worker_ordered_unmap
            assert args == ([64, 128], 17)
            return [True] * 8

    engine_mod = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.model_executor = Executor()
            self.scheduler = SimpleNamespace(
                kv_cache_manager=SimpleNamespace(
                    block_pool=SimpleNamespace(kv_cache_manager=manager)
                )
            )

    setattr(engine_mod, "EngineCore", EngineCore)
    assert patches.EngineCorePatch().patch_engine_init(engine_mod)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )

    EngineCore(config)
    manager.page_allocator.callback(4, [64, 128])

    notify.assert_called_once_with(4, -1)
    manager._increment_operation_counter.assert_called_once_with(
        "physical_growth_capacity_notifications_total"
    )


def test_ordered_unmap_does_not_publish_after_partial_failure(monkeypatch):
    patches = _load_patches(monkeypatch)
    monkeypatch.setattr(patches, "enable_kvcached", lambda: True)
    interfaces = __import__(
        "kvcached.integration.vllm.interfaces", fromlist=["init_kvcached"]
    )
    monkeypatch.setattr(interfaces, "init_kvcached", mock.Mock())
    tp_ipc_util = __import__("kvcached.tp_ipc_util", fromlist=["unused"])
    notify = mock.Mock(return_value=True)
    monkeypatch.setattr(
        tp_ipc_util, "notify_physical_growth_capacity_changed", notify
    )

    page_allocator = SimpleNamespace(callback=None)
    page_allocator.set_broadcast_unmap_callback = lambda callback: setattr(
        page_allocator, "callback", callback
    )
    manager = SimpleNamespace(
        group_id=3,
        pp_rank=0,
        page_allocator=page_allocator,
        _increment_operation_counter=mock.Mock(),
    )
    engine_mod = types.ModuleType("vllm.v1.engine.core")

    class EngineCore:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.model_executor = SimpleNamespace(
                collective_rpc=lambda method, args: [True, False]
            )
            self.scheduler = SimpleNamespace(
                kv_cache_manager=SimpleNamespace(
                    block_pool=SimpleNamespace(kv_cache_manager=manager)
                )
            )

    setattr(engine_mod, "EngineCore", EngineCore)
    assert patches.EngineCorePatch().patch_engine_init(engine_mod)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )
    EngineCore(config)

    with pytest.raises(RuntimeError, match="Ordered KV unmap failed"):
        page_allocator.callback(2, [64])

    notify.assert_not_called()
    manager._increment_operation_counter.assert_not_called()
