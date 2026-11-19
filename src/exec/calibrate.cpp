// Cost-model calibration ("hardware fingerprint"): micro-benchmarks for PCIe / launch latency, then synthetic
// queries run through the real executors on both devices at several sizes. Observed stage times become the
// training set of the per-(stage, device) linear models. The result is persisted per machine.
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "exec/engine.h"

namespace warpdb {
namespace {

// Deterministic synthetic star schema: fact f(k,y,grp,g2,x) and dimension d(k,y,w), |d| = |f| / 20.
void make_calibration_tables(Engine& e, size_t N, std::vector<uint64_t>& uids) {
  auto mk_i32 = [&](size_t n, auto gen) {
    auto c = std::make_shared<Column>();
    c->type = Type::INT32;
    c->i32.resize(n);
    for (size_t i = 0; i < n; ++i) c->i32[i] = static_cast<int32_t>(gen(i));
    uids.push_back(c->uid);
    return c;
  };
  auto mk_f64 = [&](size_t n, auto gen) {
    auto c = std::make_shared<Column>();
    c->type = Type::FLOAT64;
    c->f64.resize(n);
    for (size_t i = 0; i < n; ++i) c->f64[i] = gen(i);
    uids.push_back(c->uid);
    return c;
  };
  auto h = [](uint64_t x) {  // cheap deterministic hash -> pseudo-random numbers
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
  };
  const size_t D = std::max<size_t>(N / 20, 1);
  auto f = std::make_shared<Table>();
  f->name = "f";
  f->names = {"k", "y", "grp", "g2", "x"};
  f->cols = {mk_i32(N, [&](size_t i) { return h(i * 5 + 1) % D; }), mk_i32(N, [&](size_t i) { return h(i * 5 + 2) % 1000; }),
             mk_i32(N, [&](size_t i) { return h(i * 5 + 3) % 16; }),
             mk_i32(N, [&](size_t i) { return h(i * 5 + 4) % std::max<size_t>(N / 10, 1); }),
             mk_f64(N, [&](size_t i) { return static_cast<double>(h(i * 5 + 5) % 1000000) / 1e6; })};
  f->nrows = N;
  auto d = std::make_shared<Table>();
  d->name = "d";
  d->names = {"k", "y", "w"};
  d->cols = {mk_i32(D, [](size_t i) { return i; }), mk_i32(D, [](size_t i) { return i % 1000; }),
             mk_f64(D, [](size_t i) { return static_cast<double>(i % 97); })};
  d->nrows = D;
  e.catalog.add(f);
  e.catalog.add(d);
}

const char* kCalibrationQueries[] = {
    "select count(*) from f",
    "select count(*) from f where x < 0.5",
    "select count(*) from f where x < 0.5 and y < 500 and k > 100",
    "select count(*) from f, d where f.k = d.k",
    "select count(*) from f, d where f.k = d.k and f.y = d.y",
    "select grp, sum(x) from f group by grp",
    "select grp, g2, sum(x), avg(y), min(x), max(y) from f group by grp, g2",
    "select d.y, sum(f.x) from f, d where f.k = d.k group by d.y",
    "select k from f order by x desc limit 100",
    "select k, y from f order by grp, y limit 50",
};

}  // namespace

std::string Engine::hw_fingerprint() const {
  return (gpu_ ? gpu_->fingerprint() : std::string("cpu-only")) + "|threads=" + std::to_string(num_threads());
}

bool Engine::load_profile() {
  if (profile_path.empty()) return false;
  std::ifstream f(profile_path);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  CostModel loaded;
  if (!loaded.from_json(ss.str(), hw_fingerprint()) || !loaded.calibrated) return false;
  cost = loaded;
  return true;
}

void Engine::save_profile() const {
  if (profile_path.empty()) return;
  try {
    std::filesystem::path p(profile_path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::ofstream(profile_path) << cost.to_json();
  } catch (...) {
    // persistence is best-effort
  }
}

// A fresh process pays one-off GPU costs (context creation, lazy kernel-module loading) on its first queries.
// Those are not part of steady-state performance, so they are paid here, once, instead of inside a user query.
void Engine::warm_up_gpu() {
  if (!gpu_) return;
  Engine tmp;
  tmp.set_gpu_backend(gpu_);
  tmp.device = "gpu";
  tmp.learn = false;
  std::vector<uint64_t> uids;
  make_calibration_tables(tmp, 20000, uids);
  for (const char* q : kCalibrationQueries) tmp.execute(q);
  gpu_->forget_columns(uids);
}

void Engine::ensure_calibrated() {
  if (cost.calibrated) return;
  if (load_profile()) {
    if (on_notice) on_notice("loaded cost model from " + profile_path);
    warm_up_gpu();
    return;
  }
  calibrate();
}

void Engine::calibrate() {
  Timer total;
  if (on_notice) on_notice("calibrating the CPU/GPU cost model (one-off, ~10 s)...");
  Engine tmp;  // private catalog + private cost model; shares the GPU backend and its column cache
  tmp.set_gpu_backend(gpu_);
  tmp.cost.hw.fingerprint = hw_fingerprint();
  if (gpu_) {
    HwProfile hw;
    hw.fingerprint = tmp.cost.hw.fingerprint;
    if (gpu_->measure_hardware(hw)) tmp.cost.hw = hw;
    tmp.cost.seed_upload_from_hw();
  }
  const size_t sizes[] = {20000, 200000, 1000000};
  std::vector<uint64_t> uids;
  for (size_t N : sizes) {
    make_calibration_tables(tmp, N, uids);
    for (const char* q : kCalibrationQueries) {
      const bool devices[] = {false, true};
      for (bool on_gpu : devices) {
        if (on_gpu && !gpu_) continue;
        tmp.device = on_gpu ? "gpu" : "cpu";
        tmp.learn = false;
        tmp.execute(q);  // warm-up: uploads columns, loads kernel modules, fills allocator caches
        tmp.learn = true;
        tmp.execute(q);  // measured: the stage statistics train the model
      }
    }
    if (gpu_) gpu_->forget_columns(uids);  // calibration data must not occupy the user's VRAM cache
    uids.clear();
  }
  cost = tmp.cost;
  cost.calibrated = true;
  save_profile();
  if (on_notice) on_notice("calibration finished in " + std::to_string(static_cast<int>(total.ms())) + " ms");
}

}  // namespace warpdb
