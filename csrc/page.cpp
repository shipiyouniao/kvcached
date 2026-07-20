// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#include "page.hpp"

#include <stdexcept>
#include <string>

#include "constants.hpp"
#include "gpu_utils.hpp"

namespace kvcached {

namespace {

template <typename Status>
[[noreturn]] void throw_gpu_error(Status status, const char *operation,
                                  bool capacity_error = false) {
  std::string message = capacity_error ? "capacity_exhausted: " : "";
  message += operation;
  message += " failed in ";
  message += gpu_vmm::backend_name();
  message += ": ";
  message += gpu_vmm::error_string(status);
  throw std::runtime_error(message);
}

} // namespace

GPUPage::GPUPage(page_id_t page_id, int dev_idx, size_t page_size)
    : page_id_(page_id), dev_idx_(dev_idx),
      page_size_(page_size > 0 ? page_size : kPageSize), handle_() {
  auto prop = gpu_vmm::make_pinned_device_allocation_prop(dev_idx_);
  auto status = gpu_vmm::mem_create(&handle_, page_size_, &prop);
  if (!gpu_vmm::is_success(status)) {
    throw_gpu_error(status, "physical page allocation",
                    gpu_vmm::is_out_of_memory(status));
  }
}

GPUPage::~GPUPage() {
  auto status = gpu_vmm::mem_release(handle_);
  if (!gpu_vmm::is_success(status)) {
    LOGGER(ERROR, "physical page release failed in %s: %s",
           gpu_vmm::backend_name(), gpu_vmm::error_string(status));
  }
}

bool GPUPage::map(generic_ptr_t vaddr, bool set_access) {
  auto access_desc = gpu_vmm::make_device_rw_access_desc(dev_idx_);
  auto map_status = gpu_vmm::mem_map(vaddr, page_size_, 0, handle_);
  if (!gpu_vmm::is_success(map_status)) {
    throw_gpu_error(map_status, "physical page map");
  }
  if (set_access) {
    auto access_status =
        gpu_vmm::set_access(vaddr, page_size_, &access_desc, 1);
    if (!gpu_vmm::is_success(access_status)) {
      auto unmap_status = gpu_vmm::mem_unmap(vaddr, page_size_);
      if (!gpu_vmm::is_success(unmap_status)) {
        LOGGER(ERROR, "page map rollback failed in %s: %s",
               gpu_vmm::backend_name(), gpu_vmm::error_string(unmap_status));
      }
      throw_gpu_error(access_status, "physical page access setup");
    }
  }
  return true;
}

// TODO: finish CPUPage impl.
CPUPage::CPUPage(page_id_t page_id, size_t page_size)
    : page_id_(page_id), page_size_(page_size > 0 ? page_size : kPageSize),
      mapped_addr_(nullptr) {}

CPUPage::~CPUPage() {}

bool CPUPage::map(void *vaddr, bool set_access) {
  mapped_addr_ = vaddr;
  return true;
}

} // namespace kvcached
