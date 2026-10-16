// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
#include "gpu_budget.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kvcached {
namespace {
constexpr uint64_t kMagic = 0x4b56425544473033ULL;
constexpr size_t kInstances = 128;
constexpr size_t kPools = 1024;
constexpr size_t kName = 192;
static_assert(__atomic_always_lock_free(sizeof(uint64_t), nullptr),
              "shared budget needs lock-free aligned 64-bit publication");
uint64_t load(const uint64_t &value) {
  return __atomic_load_n(&value, __ATOMIC_SEQ_CST);
}
void publish(uint64_t &target, uint64_t value) {
  __atomic_store_n(&target, value, __ATOMIC_RELEASE);
}
uint64_t add(uint64_t a, uint64_t b) {
  if (b > UINT64_MAX - a)
    throw std::overflow_error("KV budget byte overflow");
  return a + b;
}
uint64_t sampled_capacity(uint64_t charged, int64_t free_after_margin) {
  if (free_after_margin >= 0)
    return add(charged, static_cast<uint64_t>(free_after_margin));
  // Preserve the deficit until it is combined with resident charges. Handle
  // INT64_MIN without signed negation overflow.
  const auto deficit = static_cast<uint64_t>(-(free_after_margin + 1)) + 1;
  return charged > deficit ? charged - deficit : 0;
}
std::runtime_error system_error(const char *operation) {
  return std::runtime_error(std::string(operation) + ": " +
                            std::strerror(errno));
}
int lease(int fd, short type, bool wait, off_t start = 0) {
#ifdef F_OFD_SETLK
  struct flock lock{};
  lock.l_type = type;
  lock.l_whence = SEEK_SET;
  lock.l_start = start;
  lock.l_len = 1;
  int result;
  do {
    result = fcntl(fd, wait ? F_OFD_SETLKW : F_OFD_SETLK, &lock);
  } while (result < 0 && errno == EINTR);
  return result;
#else
  (void)fd;
  (void)type;
  (void)wait;
  (void)start;
  errno = ENOTSUP;
  return -1;
#endif
}
// A separate open-file description owns the calibration lease. Closing it,
// including on process death, releases admission without a persistent flag.
// A sampler must not recursively wait on its own admission lease. Ordinary
// contenders wait in the kernel, outside the budget data mutex, then retry the
// real CAS admission. The thread-local flag owns no teardown resources.
thread_local bool inside_budget_sampler = false;
struct SamplerScope {
  bool previous = inside_budget_sampler;
  SamplerScope() { inside_budget_sampler = true; }
  ~SamplerScope() { inside_budget_sampler = previous; }
};
struct CalibrationLease {
  int fd;
  explicit CalibrationLease(const std::string &path)
      : fd(open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW)) {
    if (fd < 0)
      throw system_error("open calibration lease");
  }
  ~CalibrationLease() { close(fd); }
};
bool calibration_active(int fd) {
  struct flock lock{};
  lock.l_type = F_RDLCK;
  lock.l_whence = SEEK_SET;
  lock.l_start = 1;
  lock.l_len = 1;
  if (fcntl(fd, F_OFD_GETLK, &lock) < 0)
    throw system_error("inspect calibration lease");
  return lock.l_type != F_UNLCK;
}
uint64_t monotonic_ns() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}
} // namespace

struct GpuBudget::Shared {
  uint64_t magic;
  uint64_t budget;
  uint64_t used;
  uint64_t active;
  uint64_t version;
  uint64_t last_sample_ns;
  struct Instance {
    char name[kName];
    uint64_t limit;
    uint64_t used;
    uint64_t revision;
    uint64_t has_revision;
    uint64_t valid;
  } instances[kInstances];
  struct Pool {
    uint64_t owner;
    uint64_t instance;
    int64_t group;
    alignas(8) uint64_t charged;
    uint64_t pending_grant;
    uint64_t pending_refund;
  } pools[kPools];
};

struct GpuBudget::Locked {
  GpuBudget &budget;
  std::unique_lock<std::mutex> local;
  bool control;
  explicit Locked(GpuBudget &value, bool control_plane = false)
      : budget(value), local(value.mutex_), control(control_plane) {
    value.check_owner();
    ++value.sync_acquisitions_;
    if (control) {
      const auto started = monotonic_ns();
      int result;
      do {
        result = flock(value.fd_, LOCK_EX);
      } while (result < 0 && errno == EINTR);
      value.sync_acquire_ns_ += monotonic_ns() - started;
      if (result < 0)
        throw system_error("lock KV budget control plane");
    }
    struct stat held{}, current{};
    if (fstat(value.fd_, &held) < 0 ||
        lstat(value.path_.c_str(), &current) < 0 ||
        held.st_dev != current.st_dev || held.st_ino != current.st_ino ||
        !S_ISREG(current.st_mode)) {
      if (control)
        flock(value.fd_, LOCK_UN);
      throw std::runtime_error("KV budget segment was replaced");
    }
  }
  ~Locked() {
    if (control)
      flock(budget.fd_, LOCK_UN);
  }
};

std::shared_ptr<GpuBudget> GpuBudget::connect(const std::string &path,
                                              const Sampler &sampler) {
  return std::shared_ptr<GpuBudget>(new GpuBudget(path, sampler));
}

GpuBudget::GpuBudget(const std::string &path, const Sampler &sampler)
    : path_(path), pid_(getpid()) {
  static std::atomic<uint64_t> next_owner{1};
  owner_ = (static_cast<uint64_t>(pid_) << 32) ^
           static_cast<uint64_t>(
               std::chrono::steady_clock::now().time_since_epoch().count()) ^
           next_owner.fetch_add(1);
  if (owner_ == 0)
    owner_ = 1;
  fd_ = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0660);
  if (fd_ < 0)
    throw system_error("open KV budget");
  try {
    // Serialize connection setup separately from accounting. If an initializer
    // dies, its successor must retry exclusive epoch ownership before taking a
    // read lease; queued readers must not accept the previous epoch's header.
    if (lease(fd_, F_WRLCK, true, 2) < 0)
      throw system_error("lock KV budget initialization");
    // The first live connection starts a new accounting epoch. Sampling free
    // memory includes any old untracked allocations, so they are not credited.
    // Keep the usage lease exclusive during this one-time initialization, but
    // release the data flock before asking the GPU driver for a sample.
    bool first = lease(fd_, F_WRLCK, false) == 0;
    if (!first) {
      if (errno != EAGAIN && errno != EACCES)
        throw system_error("retain KV budget");
      if (lease(fd_, F_RDLCK, true) < 0)
        throw system_error("retain KV budget");
    }
    {
      Locked lock(*this, true);
      struct stat info{};
      if (fstat(fd_, &info) < 0)
        throw system_error("stat KV budget");
      if (first) {
        if (ftruncate(fd_, sizeof(Shared)) < 0)
          throw system_error("size KV budget");
      } else if (info.st_size != static_cast<off_t>(sizeof(Shared))) {
        throw std::runtime_error("incompatible live KV budget layout");
      }
      void *address = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd_, 0);
      if (address == MAP_FAILED)
        throw system_error("map KV budget");
      shared_ = static_cast<Shared *>(address);
      if (first)
        publish(shared_->magic, 0);
    }
    if (first) {
      const auto initial_budget = sampled_capacity(0, sampler());
      Locked lock(*this, true);
      std::memset(shared_, 0, sizeof(Shared));
      shared_->budget = initial_budget;
      shared_->last_sample_ns = monotonic_ns();
      publish(shared_->magic, kMagic);
      if (lease(fd_, F_RDLCK, false) < 0)
        throw system_error("retain initialized KV budget");
    } else {
      Locked lock(*this, true);
      if (shared_->magic != kMagic)
        throw std::runtime_error("incompatible KV budget protocol");
    }
    if (lease(fd_, F_UNLCK, false, 2) < 0)
      throw system_error("unlock KV budget initialization");
  } catch (...) {
    if (shared_)
      munmap(shared_, sizeof(Shared));
    close(fd_);
    throw;
  }
}

GpuBudget::~GpuBudget() {
  if (getpid() == pid_ && shared_) {
    try {
      Locked lock(*this, true);
      for (auto slot : slots_) {
        auto &pool = shared_->pools[slot];
        // Unknown resources remain charged while any other participant lives.
        if (pool.owner == owner_ && load(pool.charged) == 0 &&
            load(pool.pending_grant) == 0 && load(pool.pending_refund) == 0)
          publish(pool.owner, 0);
      }
      reclaim_instances_locked();
    } catch (const std::exception &error) {
      std::cerr << "KV budget teardown: " << error.what() << '\n';
    }
  }
  if (shared_)
    munmap(shared_, sizeof(Shared));
  if (fd_ >= 0)
    close(fd_);
}

void GpuBudget::check_owner() const {
  if (getpid() != pid_)
    throw std::runtime_error("reconnect KV budget after fork");
}

void GpuBudget::reclaim_instances_locked() {
  bool referenced[kInstances]{};
  for (const auto &pool : shared_->pools) {
    // Retain limits for live participants and for resources whose release has
    // not been confirmed, including resources left behind by a dead owner.
    if (pool.owner != 0 || load(pool.charged) != 0) {
      if (pool.instance >= kInstances)
        throw std::runtime_error("invalid KV budget instance reference");
      referenced[pool.instance] = true;
    }
  }
  for (size_t i = 0; i < kInstances; ++i) {
    if (!referenced[i])
      publish(shared_->instances[i].valid, 0);
  }
}

size_t GpuBudget::register_pool(const std::string &instance, int64_t group) {
  if (instance.empty() || instance.size() >= kName ||
      instance.find('\0') != std::string::npos)
    throw std::invalid_argument(
        "KV budget instance name must contain 1..191 bytes");
  Locked lock(*this, true);
  // Also recover an instance published before pool registration was
  // interrupted.
  reclaim_instances_locked();
  size_t instance_index = kInstances;
  for (size_t i = 0; i < kInstances; ++i) {
    if (shared_->instances[i].valid && instance == shared_->instances[i].name) {
      instance_index = i;
      break;
    }
  }
  if (instance_index == kInstances) {
    for (size_t i = 0; i < kInstances; ++i) {
      auto &entry = shared_->instances[i];
      if (!entry.valid) {
        std::memset(&entry, 0, sizeof(entry));
        std::memcpy(entry.name, instance.c_str(), instance.size() + 1);
        entry.limit = UINT64_MAX;
        publish(entry.valid, 1);
        instance_index = i;
        break;
      }
    }
  }
  if (instance_index == kInstances)
    throw std::runtime_error("KV budget instance table is full");
  for (auto slot : slots_) {
    const auto &entry = shared_->pools[slot];
    if (entry.instance == instance_index && entry.group == group)
      return slot;
  }
  for (size_t i = 0; i < kPools; ++i) {
    auto &entry = shared_->pools[i];
    if (entry.owner == 0 && load(entry.charged) == 0 &&
        load(entry.pending_grant) == 0 && load(entry.pending_refund) == 0) {
      slots_.push_back(i); // allocate before publishing ownership
      entry.instance = instance_index;
      entry.group = group;
      publish(entry.owner, owner_);
      return i;
    }
  }
  throw std::runtime_error("KV budget pool table is full");
}

bool GpuBudget::reserve_counter(uint64_t &used, const uint64_t &limit,
                                uint64_t bytes) {
  const auto started = monotonic_ns();
  uint64_t waiting = 0;
  auto finish = [&] {
    const auto ended = monotonic_ns();
    sync_acquire_ns_ += ended - started;
    if (waiting)
      sync_wait_ns_ += ended - waiting;
  };
  uint64_t expected = load(used);
  for (;;) {
    const auto cap = load(limit);
    if (expected > cap || bytes > cap - expected) {
      finish();
      return false;
    }
    const auto next = expected + bytes;
    if (__atomic_compare_exchange_n(&used, &expected, next, false,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
      finish();
      // A limit shrink can race the first read. A completed reservation may
      // precede a later shrink; a shrink already visible here must be honored.
      if (next > load(limit)) {
        subtract_counter(used, bytes);
        return false;
      }
      return true;
    }
    if (!waiting) {
      waiting = monotonic_ns();
      ++sync_contentions_;
    }
    ++sync_cas_retries_;
  }
}

void GpuBudget::subtract_counter(uint64_t &used, uint64_t bytes) {
  if (!bytes)
    return;
  const auto started = monotonic_ns();
  uint64_t waiting = 0;
  uint64_t expected = load(used);
  for (;;) {
    if (bytes > expected)
      throw std::runtime_error("state_inconsistency: CAS budget underflow");
    if (__atomic_compare_exchange_n(&used, &expected, expected - bytes, false,
                                    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
      break;
    if (!waiting) {
      waiting = monotonic_ns();
      ++sync_contentions_;
    }
    ++sync_cas_retries_;
  }
  const auto ended = monotonic_ns();
  sync_acquire_ns_ += ended - started;
  if (waiting)
    sync_wait_ns_ += ended - waiting;
}

void GpuBudget::accounting_test_hook(int stage) {
#ifdef KVCACHED_BUDGET_TESTING
  if (accounting_test_hook_)
    accounting_test_hook_(stage);
#else
  (void)stage;
#endif
}

BudgetSnapshot GpuBudget::snapshot_locked(size_t slot) const {
  if (slot >= kPools || load(shared_->pools[slot].owner) != owner_)
    throw std::runtime_error("invalid KV budget owner");
  BudgetSnapshot result;
  const auto instance = shared_->pools[slot].instance;
  result.budget_bytes = load(shared_->budget);
  result.charged_bytes = load(shared_->used);
  result.instance_charged_bytes = load(shared_->instances[instance].used);
  result.instance_limit_bytes = load(shared_->instances[instance].limit);
  result.active_mutations = load(shared_->active);
  result.sync_acquisitions = sync_acquisitions_;
  result.sync_contentions = sync_contentions_;
  result.sync_acquire_ns = sync_acquire_ns_;
  result.sync_wait_ns = sync_wait_ns_;
  result.sync_cas_retries = sync_cas_retries_;
  result.sync_dead_owner_recoveries = sync_dead_owner_recoveries_;
  return result;
}

BudgetSnapshot GpuBudget::snapshot(size_t slot, const Sampler &sampler) {
  bool sample_due = false;
  if (sampler) {
    Locked lock(*this);
    constexpr uint64_t interval_ns = 1000000000ULL;
    sample_due = !load(shared_->active) &&
                 monotonic_ns() - load(shared_->last_sample_ns) >= interval_ns;
  }
  // Admission queries also refresh an exhausted budget: waiting for another
  // grant would otherwise prevent a scheduler from observing recovered memory.
  if (sample_due)
    recalibrate(sampler);
  Locked lock(*this);
  return snapshot_locked(slot);
}

std::unique_ptr<BudgetGrant> GpuBudget::grant(size_t slot, uint64_t bytes,
                                              BudgetSnapshot *observed,
                                              const Sampler &sampler) {
  auto result = std::make_unique<BudgetGrant>(shared_from_this(), slot, 0);
  bool attempted_sample = false;
  for (;;) {
    bool sample_due = false, wait_for_calibration = false;
    {
      Locked lock(*this);
      constexpr uint64_t interval_ns = 1000000000ULL;
      sample_due =
          sampler && !attempted_sample && !load(shared_->active) &&
          monotonic_ns() - load(shared_->last_sample_ns) >= interval_ns;
      if (!sample_due) {
        const auto state = snapshot_locked(slot);
        if (observed)
          *observed = state;
        auto &pool = shared_->pools[slot];
        auto &instance = shared_->instances[pool.instance];
        const auto charge = add(load(pool.charged), bytes);
        // Intent precedes either CAS. An interrupted reservation remains tied
        // to its owner, even if only one aggregate counter was updated.
        publish(pool.pending_grant, bytes);
        __atomic_add_fetch(&shared_->active, 1, __ATOMIC_SEQ_CST);
        bool admitted = false;
        bool instance_held = false, gpu_held = false, owner_published = false;
        try {
          // The increment-before-lease-check handshake prevents calibration
          // from sampling an unmaterialized promise as free physical memory.
          if (calibration_active(fd_)) {
            wait_for_calibration = !inside_budget_sampler;
            throw std::runtime_error(
                "capacity_exhausted: KV budget calibration in progress");
          }
          __atomic_add_fetch(&shared_->version, 1, __ATOMIC_SEQ_CST);
          admitted = true;
          accounting_test_hook(1);
          instance_held = reserve_counter(instance.used, instance.limit, bytes);
          if (!instance_held)
            throw std::runtime_error(
                "capacity_exhausted: instance KV budget would be exceeded");
          accounting_test_hook(2);
          gpu_held = reserve_counter(shared_->used, shared_->budget, bytes);
          if (!gpu_held)
            throw std::runtime_error(
                "capacity_exhausted: shared KV budget would be exceeded"
                " (requested=" +
                std::to_string(bytes) +
                ", budget=" + std::to_string(load(shared_->budget)) +
                ", charged=" + std::to_string(load(shared_->used)) + ")");
          accounting_test_hook(3);
          publish(pool.charged, charge);
          owner_published = true;
          accounting_test_hook(4);
          publish(pool.pending_grant, 0);
#ifdef KVCACHED_BUDGET_TESTING
          if (grant_test_hook_)
            grant_test_hook_();
#endif
          result->unused_ = bytes;
          result->active_ = true;
          return result;
        } catch (...) {
          if (owner_published)
            subtract_counter(pool.charged, bytes);
          if (gpu_held)
            subtract_counter(shared_->used, bytes);
          if (instance_held)
            subtract_counter(instance.used, bytes);
          publish(pool.pending_grant, 0);
          if (admitted)
            __atomic_add_fetch(&shared_->version, 1, __ATOMIC_SEQ_CST);
          __atomic_sub_fetch(&shared_->active, 1, __ATOMIC_SEQ_CST);
          if (!wait_for_calibration)
            throw;
        }
      }
    }
    if (wait_for_calibration) {
      accounting_test_hook(40);
      CalibrationLease waiter(path_);
      if (lease(waiter.fd, F_RDLCK, true, 1) < 0)
        throw system_error("wait for KV budget calibration");
      continue;
    }
    attempted_sample = true;
    recalibrate(sampler);
  }
}

std::string GpuBudget::set_instance_limit(size_t slot, uint64_t bytes,
                                          uint64_t revision) {
  Locked lock(*this, true);
  (void)snapshot_locked(slot);
  auto &entry = shared_->instances[shared_->pools[slot].instance];
  if (entry.has_revision && revision < entry.revision)
    return "stale";
  if (entry.has_revision && revision == entry.revision)
    return entry.limit == bytes ? "applied" : "conflict";
  publish(entry.limit, bytes);
  entry.revision = revision;
  entry.has_revision = 1;
  __atomic_add_fetch(&shared_->version, 1, __ATOMIC_SEQ_CST);
  return "applied";
}

bool GpuBudget::recalibrate(const Sampler &sampler) {
  uint64_t charged = 0, version = 0;
  CalibrationLease calibration(path_);
  {
    Locked lock(*this, true);
    struct stat held{}, selected{};
    if (fstat(fd_, &held) < 0 || fstat(calibration.fd, &selected) < 0 ||
        held.st_dev != selected.st_dev || held.st_ino != selected.st_ino)
      throw std::runtime_error("KV budget calibration segment was replaced");
    if (lease(calibration.fd, F_WRLCK, false, 1) < 0) {
      if (errno == EAGAIN || errno == EACCES)
        return false;
      throw system_error("lock calibration lease");
    }
    if (load(shared_->active))
      return false;
    accounting_test_hook(20);
    version = load(shared_->version);
    charged = load(shared_->used);
  }
  SamplerScope sampling;
  const int64_t free_budget = sampler();
  Locked lock(*this, true);
  if (load(shared_->active) || load(shared_->version) != version)
    return false;
  // New grants still cannot pass the lease check. A release starting after
  // this validation cannot invalidate the sample taken before its CUDA work.
  // The spendable credit is budget - used, so this restores exactly the
  // sampled free capacity, not the charged bytes a second time. Abandoned
  // idle-owner charges remain in both terms and cancel; they are never
  // refunded or reclaimed online. Interrupted mutations fail the checks above.
  // A later valid refund follows physical release and increases real free
  // capacity by the same amount. Keep negative headroom in sampled_capacity.
  publish(shared_->budget, sampled_capacity(charged, free_budget));
  publish(shared_->last_sample_ns, monotonic_ns());
  __atomic_add_fetch(&shared_->version, 1, __ATOMIC_SEQ_CST);
  return true;
}

void GpuBudget::begin_mutation() {
  check_owner();
  std::lock_guard<std::mutex> mutation_lock(mutation_mutex_);
  const auto thread = std::this_thread::get_id();
  auto &local = mutations_[thread];
  if (local.depth++ != 0)
    return;
  try {
    Locked lock(*this);
    __atomic_add_fetch(&shared_->active, 1, __ATOMIC_SEQ_CST);
    __atomic_add_fetch(&shared_->version, 1, __ATOMIC_SEQ_CST);
  } catch (...) {
    mutations_.erase(thread);
    throw;
  }
}

void GpuBudget::return_locked(size_t slot, uint64_t bytes) {
  if (slot >= kPools || load(shared_->pools[slot].owner) != owner_ ||
      bytes > load(shared_->pools[slot].charged))
    throw std::runtime_error("state_inconsistency: invalid KV budget refund");
  if (!bytes)
    return;
  auto &pool = shared_->pools[slot];
  auto &instance = shared_->instances[pool.instance];
  // Refund follows confirmed release or an unused reservation. Publish the
  // intent first and drop owner charge before aggregates, so interruption can
  // only leave aggregate credit withheld, never grant it twice.
  publish(pool.pending_refund, bytes);
  accounting_test_hook(10);
  subtract_counter(pool.charged, bytes);
  accounting_test_hook(11);
  subtract_counter(instance.used, bytes);
  accounting_test_hook(12);
  subtract_counter(shared_->used, bytes);
  accounting_test_hook(13);
  publish(pool.pending_refund, 0);
}

void GpuBudget::refund(size_t slot, uint64_t bytes) {
  if (!bytes)
    return;
  check_owner();
  {
    std::lock_guard<std::mutex> mutation_lock(mutation_mutex_);
    const auto found = mutations_.find(std::this_thread::get_id());
    if (found != mutations_.end() && found->second.depth) {
      auto &total = found->second.refunds[slot];
      total = add(total, bytes);
      return;
    }
  }
  Locked lock(*this);
  __atomic_add_fetch(&shared_->active, 1, __ATOMIC_SEQ_CST);
  __atomic_add_fetch(&shared_->version, 1, __ATOMIC_SEQ_CST);
  return_locked(slot, bytes);
  __atomic_add_fetch(&shared_->version, 1, __ATOMIC_SEQ_CST);
  __atomic_sub_fetch(&shared_->active, 1, __ATOMIC_SEQ_CST);
}

void GpuBudget::end_mutation() {
  check_owner();
  std::lock_guard<std::mutex> mutation_lock(mutation_mutex_);
  auto found = mutations_.find(std::this_thread::get_id());
  if (found == mutations_.end() || !found->second.depth)
    throw std::runtime_error("unbalanced KV budget mutation");
  if (found->second.depth > 1) {
    --found->second.depth;
    return;
  }
  Locked lock(*this);
  if (!load(shared_->active))
    throw std::runtime_error("unbalanced shared KV mutation");
  for (const auto &[slot, bytes] : found->second.refunds) {
    if (slot >= kPools || load(shared_->pools[slot].owner) != owner_ ||
        bytes > load(shared_->pools[slot].charged))
      throw std::runtime_error("state_inconsistency: invalid KV budget refund");
  }
  for (const auto &[slot, bytes] : found->second.refunds)
    return_locked(slot, bytes);
  mutations_.erase(found);
  // Publish completion before active can reach zero. Calibration must never
  // observe a completed release with only its begin-version visible.
  __atomic_add_fetch(&shared_->version, 1, __ATOMIC_SEQ_CST);
  __atomic_sub_fetch(&shared_->active, 1, __ATOMIC_SEQ_CST);
  accounting_test_hook(30);
}

BudgetReceipt::BudgetReceipt(std::shared_ptr<GpuBudget> budget, size_t slot,
                             uint64_t bytes)
    : budget_(std::move(budget)), slot_(slot), bytes_(bytes) {}
void BudgetReceipt::release() {
  if (bytes_) {
    budget_->refund(slot_, bytes_);
    bytes_ = 0;
  }
}
BudgetGrant::BudgetGrant(std::shared_ptr<GpuBudget> budget, size_t slot,
                         uint64_t bytes)
    : budget_(std::move(budget)), slot_(slot), unused_(bytes),
      active_(bytes != 0) {}
BudgetGrant::~BudgetGrant() {
  try {
    finish();
  } catch (const std::exception &error) {
    std::cerr << "KV grant cleanup remains charged: " << error.what() << '\n';
  }
}
std::unique_ptr<BudgetReceipt> BudgetGrant::take(uint64_t bytes) {
  if (!active_ || bytes > unused_)
    throw std::runtime_error("invalid KV grant consumption");
  auto result = std::make_unique<BudgetReceipt>(budget_, slot_, bytes);
  unused_ -= bytes;
  return result;
}
void BudgetGrant::finish() {
  if (!active_)
    return;
  GpuBudget::Locked lock(*budget_);
  budget_->return_locked(slot_, unused_);
  unused_ = 0;
  if (!load(budget_->shared_->active))
    throw std::runtime_error("unbalanced KV grant");
  __atomic_add_fetch(&budget_->shared_->version, 1, __ATOMIC_SEQ_CST);
  __atomic_sub_fetch(&budget_->shared_->active, 1, __ATOMIC_SEQ_CST);
  active_ = false;
}
BudgetMutation::BudgetMutation(std::shared_ptr<GpuBudget> budget)
    : budget_(std::move(budget)) {
  if (budget_)
    budget_->begin_mutation();
}
BudgetMutation::~BudgetMutation() {
  try {
    finish();
  } catch (const std::exception &error) {
    std::cerr << "KV mutation cleanup remains charged: " << error.what()
              << '\n';
  }
}
void BudgetMutation::finish() {
  if (active_ && budget_)
    budget_->end_mutation();
  active_ = false;
}
} // namespace kvcached
