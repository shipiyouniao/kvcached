# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from collections import deque
from importlib.machinery import ModuleSpec
from types import SimpleNamespace
from unittest import mock


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
