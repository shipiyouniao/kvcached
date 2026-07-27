# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Python boundary for memory information providers."""

from __future__ import annotations

from kvcached.utils import (
    MEMINFO_PROVIDER,
    MEMINFO_TIMEOUT_MS,
    KVCachedConfigError,
)


def query_mem_info(world_size: int, pp_rank: int) -> tuple[int, int]:
    """Return worker-visible free and total GPU memory in bytes."""
    if MEMINFO_PROVIDER != "worker":
        raise KVCachedConfigError(
            "KVCACHED_ENGINECORE_NO_CUDA=true requires KVCACHED_MEMINFO_PROVIDER=worker"
        )

    from kvcached.tp_ipc_util import query_worker_cuda_mem_get_info

    del world_size
    effective_pp_rank = pp_rank if pp_rank >= 0 else 0
    return query_worker_cuda_mem_get_info(
        0,
        effective_pp_rank,
        timeout=max(MEMINFO_TIMEOUT_MS, 1) / 1000.0,
    )
