# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types

import pytest

if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
    sys.modules.setdefault("torch", types.ModuleType("torch"))

from kvcached import meminfo_provider


def test_worker_provider_queries_normalized_pipeline_rank(monkeypatch):
    observed = {}

    def query_worker(rank, pp_rank, timeout):
        observed["args"] = (rank, pp_rank, timeout)
        return 123, 456

    monkeypatch.setattr(meminfo_provider, "MEMINFO_PROVIDER", "worker")
    monkeypatch.setattr(meminfo_provider, "MEMINFO_TIMEOUT_MS", 250)
    monkeypatch.setattr(
        "kvcached.tp_ipc_util.query_worker_cuda_mem_get_info",
        query_worker,
    )

    assert meminfo_provider.query_mem_info(4, -1) == (123, 456)
    assert observed["args"] == (0, 0, 0.25)


def test_non_worker_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(meminfo_provider, "MEMINFO_PROVIDER", "local")

    with pytest.raises(
        meminfo_provider.KVCachedConfigError,
        match="requires KVCACHED_MEMINFO_PROVIDER=worker",
    ):
        meminfo_provider.query_mem_info(1, 0)
