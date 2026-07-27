# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

import os

import pytest


def _compiled_vmm_ops():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for VMM transaction validation")

    try:
        from kvcached import vmm_ops
    except ImportError:
        pytest.skip("kvcached.vmm_ops extension is not built")
    required_operations = (
        "init_kvcached",
        "create_kv_tensors",
        "map_to_kv_tensors_with_result",
        "unmap_from_kv_tensors",
        "shutdown_kvcached",
    )
    if any(not hasattr(vmm_ops, name) for name in required_operations):
        pytest.skip("a test stub replaced the compiled kvcached.vmm_ops module")
    return torch, vmm_ops


def test_cuda_map_batch_rolls_back_new_pages_and_preserves_idempotency():
    _torch, vmm_ops = _compiled_vmm_ops()

    page_size = 2 * 1024 * 1024
    vmm_ops.init_kvcached("cuda:0", page_size, False)
    try:
        vmm_ops.create_kv_tensors(
            8 * 1024 * 1024,
            2,
            "cuda:0",
            1,
            2,
            0,
            False,
        )

        with pytest.raises(RuntimeError, match="zero page unmap failed"):
            vmm_ops.map_to_kv_tensors_with_result(
                [0, 16 * 1024 * 1024],
                0,
            )

        # Offset zero was mapped before the invalid second offset failed. It
        # must be new again after the failed batch rolls back.
        assert vmm_ops.map_to_kv_tensors_with_result([0], 0) == (True, [0])
        assert vmm_ops.map_to_kv_tensors_with_result([0], 0) == (True, [])
        assert vmm_ops.unmap_from_kv_tensors([0], 0)
        assert vmm_ops.unmap_from_kv_tensors([0], 0)
    finally:
        vmm_ops.shutdown_kvcached()


def test_page_allocator_rebinds_the_calling_thread_to_its_device():
    torch, vmm_ops = _compiled_vmm_ops()
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for device rebinding validation")

    torch.cuda.set_device(0)
    allocator = vmm_ops.PageAllocator(
        2,
        8 * 1024 * 1024,
        2 * 1024 * 1024,
        1,
        0,
        False,
        False,
        False,
        2,
        92,
        f"device-binding-test-{os.getpid()}",
    )

    torch.cuda.set_device(1)
    assert torch.cuda.current_device() == 1
    assert allocator.get_avail_physical_pages() > 0
    assert torch.cuda.current_device() == 0
