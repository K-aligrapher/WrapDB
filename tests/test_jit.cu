// JIT pipeline fusion: parity with the CPU engine and the interpreted GPU kernels, cache behaviour, codegen limits.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>

#include "exec/engine.h"
#include "gpu/gpu_backend.h"
#include "gpu/jit/codegen.h"
#include "gpu/jit/jit.h"
#include "storage/csv.h"

using namespace warpdb;
using warpdb::gpu::gen_agg_plan;
using warpdb::gpu::gen_filter_plan;
using warpdb::gpu::JAgg;
using warpdb::gpu::JAggDesc;
using warpdb::gpu::JitError;
using warpdb::gpu::JitPlan;
using warpdb::gpu::JitRuntime;
using warpdb::gpu::JitUnsupported;

namespace {

std::string write_tmp(const std::string& name, const std::string& body) {
  std::string path = (std::filesystem::temp_directory_path() / name).string();
  std::ofstream(path, std::ios::binary) << body;
  return path;
}

void make_data(Engine& eng) {
  std::mt19937 g(4321);
  std::string fact = "id,cust,prod,qty,price,day,flag,note\n";
  const char* flags[] = {"A", "B", "C"};
  const char* notes[] = {"alpha", "beta", "gamma", "delta", "epsilon"};
  for (int i = 0; i < 30000; ++i) {
    int day = 18000 + static_cast<int>(g() % 900);
    int y;
    unsigned m, d;
    civil_from_days(day, y, m, d);
    char buf[200];
    std::snprintf(buf, sizeof buf, "%d,%d,%d,%d,%.2f,%04d-%02u-%02u,%s,%s\n", i, static_cast<int>(g() % 600),
                  static_cast<int>(g() % 40), 1 + static_cast<int>(g() % 50), (g() % 100000) / 100.0, y, m, d,
                  flags[g() % 3], notes[g() % 5]);
    fact += buf;
  }
  std::string cust = "cid,region,seg\n";
  const char* regions[] = {"ASIA", "EUROPE", "AMERICA", "AFRICA"};
  for (int c = 0; c < 600; ++c) cust += std::to_string(c) + "," + regions[c % 4] + ",seg" + std::to_string(c % 7) + "\n";
  std::string prod = "pid,cat,name\n";
  for (int p = 0; p < 40; ++p)
    prod += std::to_string(p) + ",cat" + std::to_string(p % 5) + ",item " + std::to_string(p) + (p % 3 ? " red" : " blue") + "\n";
  eng.catalog.add(load_csv(write_tmp("wdb_jfact.csv", fact), "fact"));
  eng.catalog.add(load_csv(write_tmp("wdb_jcust.csv", cust), "cust"));
  eng.catalog.add(load_csv(write_tmp("wdb_jprod.csv", prod), "prod"));
}

std::vector<std::vector<std::string>> rows_of(const QueryResult& r, bool sort_rows) {
  std::vector<std::vector<std::string>> out(r.nrows);
  for (size_t i = 0; i < r.nrows; ++i)
    for (auto& c : r.cols) out[i].push_back(format_cell(*c, i));
  if (sort_rows) std::sort(out.begin(), out.end());
  return out;
}

bool cells_equal(const std::string& a, const std::string& b) {
  if (a == b) return true;
  char *ea, *eb;
  double x = std::strtod(a.c_str(), &ea), y = std::strtod(b.c_str(), &eb);
  if (*ea || *eb) return false;
  return std::fabs(x - y) <= 1e-9 * std::max(1.0, std::max(std::fabs(x), std::fabs(y)));
}

bool trace_has(const QueryResult& r, const std::string& what) {
  for (auto& t : r.trace)
    if (t.find(what) != std::string::npos) return true;
  return false;
}

struct Jit : ::testing::Test {
  static Engine& eng() {
    static Engine e;
    static bool init = false;
    if (!init) {
      make_data(e);
      auto be = gpu::make_gpu_backend();
      e.set_gpu_backend(be);
      be->set_jit_min_rows(0);  // fuse even for these small inputs
      JitRuntime::instance().set_disk_cache_dir((std::filesystem::temp_directory_path() / "wdb_jit_test_cache").string());
      init = true;
    }
    return e;
  }
  static gpu::GpuBackend& backend() { return *dynamic_cast<gpu::GpuBackend*>(eng().gpu_backend()); }

  // CPU result == fused GPU result == interpreted GPU result
  void expect_same(const std::string& sql, bool also_interpreted = true) {
    Engine& e = eng();
    bool ordered = sql.find("order by") != std::string::npos;
    e.device = "cpu";
    auto cpu = rows_of(e.execute(sql), !ordered);
    e.device = "gpu";
    backend().set_jit(true);
    QueryResult fused_r = e.execute(sql);
    auto fused = rows_of(fused_r, !ordered);
    auto check = [&](const std::vector<std::vector<std::string>>& got, const char* who) {
      ASSERT_EQ(cpu.size(), got.size()) << who << ": " << sql;
      for (size_t i = 0; i < cpu.size(); ++i) {
        ASSERT_EQ(cpu[i].size(), got[i].size());
        for (size_t j = 0; j < cpu[i].size(); ++j)
          ASSERT_TRUE(cells_equal(cpu[i][j], got[i][j])) << who << ": " << sql << "\n row " << i << " col " << j
                                                           << ": cpu=" << cpu[i][j] << " got=" << got[i][j];
      }
    };
    check(fused, "jit");
    if (also_interpreted) {
      backend().set_jit(false);
      auto interp = rows_of(e.execute(sql), !ordered);
      check(interp, "interpreted gpu");
      backend().set_jit(true);
    }
    e.device = "cpu";
  }
};

TEST_F(Jit, RuntimeIsAvailable) { EXPECT_TRUE(JitRuntime::instance().available()); }

TEST_F(Jit, FusedFilterMatchesAcrossPredicateShapes) {
  expect_same("select count(*) from fact where qty > 25");
  expect_same("select count(*), sum(qty * price) from fact where qty between 10 and 20 and price < 500.5");
  expect_same("select count(*) from fact where flag = 'B' or (qty < 5 and not flag = 'A')");
  expect_same("select count(*) from fact where flag in ('A','C') and day >= date '2019-06-01' - interval '1' month");
  expect_same("select count(*) from fact where -qty < -40 and abs(price - 500) < 10");
  expect_same("select count(*) from fact where qty % 7 = 3 and note like '%a%'");
  expect_same("select count(*) from fact where note not like 'e%' and flag <> 'A'");
  expect_same("select count(*) from fact where flag > 'A' and flag <= 'B'");
  expect_same("select count(*) from fact where cust in (1, 7, 22, 100) or prod = 5");
  expect_same("select count(*) from fact where 1 = 0");  // constant predicate
  expect_same("select count(*) from fact where price > 99999");  // nothing passes
}

TEST_F(Jit, FusedAggregateMatchesAcrossAggregateShapes) {
  expect_same("select flag, count(*), sum(qty), avg(price), min(price), max(qty) from fact group by flag order by flag");
  expect_same("select cust, sum(price) as p, count(*) as c from fact group by cust order by p desc, cust limit 20");
  expect_same("select cust, prod, count(*) from fact group by cust, prod order by cust, prod limit 50");
  expect_same("select extract(year from day) as y, extract(month from day) as m, count(*) from fact group by y, m order by y, m");
  expect_same("select flag, max(day), min(day), max(note), min(note) from fact group by flag order by flag");
  expect_same("select prod from fact group by prod having count(*) > 760 order by prod");
  expect_same("select sum(price) / count(*), max(price) - min(price), avg(qty) from fact");
  expect_same("select count(*), sum(price), avg(qty), min(qty) from fact where qty > 1000");  // empty input
  expect_same("select flag, note, sum(case when qty > 25 then price else 0 end), "
              "sum(case when flag = 'A' then 1 else 0 end) from fact group by flag, note order by flag, note");
  expect_same("select cust % 7 as g, count(*), sum(qty * price - 1) from fact group by g order by g");
  expect_same("select price, count(*) from fact where qty < 3 group by price order by price limit 30");  // double keys
}

TEST_F(Jit, FusedPipelinesAfterJoins) {
  expect_same("select region, count(*), sum(price) from fact, cust where cust = cid group by region order by region");
  expect_same("select cat, count(*) from fact join prod on prod = pid where qty > 10 group by cat order by cat");
  expect_same("select region, cat, sum(qty), avg(price) from fact, cust, prod where cust = cid and prod = pid and flag = 'A' "
              "group by region, cat order by region, cat");
  expect_same("select count(*) from cust c1, cust c2 where c1.seg = c2.seg and c1.cid < c2.cid");
  expect_same("select count(*) from fact, prod where prod = pid and name like 'item 1%'");
  expect_same("select seg, sum(price) from fact, cust where cust = cid and region = 'ASIA' and price > 100 "
              "group by seg having sum(price) > 1000 order by seg");
}

TEST_F(Jit, FusedKernelsAreActuallyUsedAndFallBackCleanly) {
  Engine& e = eng();
  e.device = "gpu";
  backend().set_jit(true);
  QueryResult single = e.execute("select flag, sum(price) from fact where qty > 3 group by flag");
  EXPECT_TRUE(trace_has(single, "jit-fused with scan+filter")) << "single-table aggregate must be one kernel";
  QueryResult joined = e.execute("select region, count(*) from fact, cust where cust = cid and qty > 3 group by region");
  EXPECT_TRUE(trace_has(joined, "gpu scan (jit-fused) fact"));
  EXPECT_TRUE(trace_has(joined, "gpu hash aggregate (jit-fused)"));
  // unsupported shape (comparing VARCHAR columns of different dictionaries): must silently use the interpreter
  QueryResult odd = e.execute("select count(*) from cust c1, cust c2 where c1.seg = c2.seg and c1.region <> c2.region");
  EXPECT_EQ(odd.nrows, 1u);
  backend().set_jit(false);
  QueryResult off = e.execute("select flag, sum(price) from fact where qty > 3 group by flag");
  EXPECT_FALSE(trace_has(off, "jit-fused"));
  backend().set_jit(true);
  e.device = "cpu";
}

// Regression: the CUDA driver API uses the calling thread's context. Hosts like Streamlit run every request on a
// different thread, and the JIT used to fail there with "invalid device context" (after wasting a full NVRTC compile).
TEST_F(Jit, WorksFromAnyThread) {
  Engine& e = eng();
  e.device = "gpu";
  backend().set_jit(true);
  const char* q = "select flag, note, count(*), sum(price * 3.25) from fact where qty > 7 group by flag, note";
  e.execute(q);  // main thread
  QueryResult from_thread;
  std::thread t([&] {  // a fresh thread that has never touched CUDA; new constant => a kernel that must be built here
    from_thread = e.execute("select flag, note, count(*), sum(price * 4.75) from fact where qty > 9 group by flag, note");
  });
  t.join();
  e.device = "cpu";
  EXPECT_TRUE(trace_has(from_thread, "jit-fused")) << "JIT must work on any thread";
  EXPECT_FALSE(trace_has(from_thread, "fell back")) << "no fallback expected";
}

TEST_F(Jit, FailedCompilesDoNotPolluteStageTimes) {
  // An unsupported shape is rejected by the code generator before any compile; the stage time must stay small and the
  // query must still be correct.
  Engine& e = eng();
  e.device = "gpu";
  QueryResult r = e.execute("select count(*) from cust c1, cust c2 where c1.seg = c2.seg and c1.region <> c2.region");
  e.device = "cpu";
  EXPECT_EQ(r.nrows, 1u);
  for (auto& st : r.stages) EXPECT_LT(st.ms, 2000.0);
}

TEST_F(Jit, GroupTableOverflowIsDetectedAndRetried) {
  Engine& e = eng();
  e.device = "cpu";
  auto cpu = rows_of(e.execute("select id, sum(qty), count(*) from fact group by id order by id"), false);  // 30000 groups
  e.device = "gpu";
  backend().set_jit(true);
  backend().set_jit_force_capacity_for_test(16);  // start far too small
  QueryResult r = e.execute("select id, sum(qty), count(*) from fact group by id order by id");
  backend().set_jit_force_capacity_for_test(0);
  auto got = rows_of(r, false);
  ASSERT_EQ(cpu.size(), got.size());
  EXPECT_EQ(cpu, got);
  EXPECT_TRUE(trace_has(r, "jit-fused"));
  e.device = "cpu";
}

TEST_F(Jit, KernelsAreCachedInMemoryAndOnDisk) {
  Engine& e = eng();
  e.device = "gpu";
  backend().set_jit(true);
  const char* q = "select flag, note, sum(price * 1.5), max(qty) from fact where price > 12.5 group by flag, note";
  e.execute(q);  // may compile
  auto s0 = JitRuntime::instance().stats();
  e.execute(q);  // same kernel text: in-memory hit, no compile
  auto s1 = JitRuntime::instance().stats();
  EXPECT_EQ(s1.compiles, s0.compiles);
  EXPECT_GT(s1.mem_hits, s0.mem_hits);
  JitRuntime::instance().clear_memory_cache();  // a "new process": the CUBIN is still on disk
  e.execute(q);
  auto s2 = JitRuntime::instance().stats();
  EXPECT_EQ(s2.compiles, s1.compiles) << "disk cache must avoid NVRTC";
  EXPECT_GT(s2.disk_hits, s1.disk_hits);
  e.device = "cpu";
}

// ------------------------------------------------------------------------------------------------ codegen
TEST_F(Jit, CodegenIsDeterministicAndRejectsUnsupportedShapes) {
  Engine& e = eng();
  PlannedQuery q = e.plan("select flag, sum(price * (1 - 0.1)) from fact where qty > 3 and note like 'a%' group by flag");
  std::vector<JAggDesc> descs;
  for (auto& a : q.aggs) {
    JAggDesc d;
    d.func = JAgg::Sum;
    d.arg = a->args[0];
    descs.push_back(d);
  }
  JitPlan a = gen_agg_plan(q, {0}, true, q.push[0], q.sel.group_by, descs);
  JitPlan b = gen_agg_plan(q, {0}, true, q.push[0], q.sel.group_by, descs);
  EXPECT_EQ(a.source, b.source);  // stable text => stable cache key
  EXPECT_EQ(a.K, 1);
  EXPECT_EQ(a.A, 1);
  EXPECT_FALSE(a.masks.empty());  // the LIKE became a dictionary mask
  EXPECT_NE(a.source.find("group_slot<1>"), std::string::npos);
  // 1 - 0.1 is folded on the host (as the interpreter does) and printed with round-trip precision
  EXPECT_NE(a.source.find("(0.9000000000"), std::string::npos) << a.source;
  EXPECT_EQ(a.source.find("0.1"), std::string::npos) << "the un-folded subexpression must not reach the kernel";

  // SUM over a VARCHAR column is not meaningful: the generator must refuse, not emit broken code
  PlannedQuery bad = e.plan("select sum(flag) from fact");
  JAggDesc bd;
  bd.func = JAgg::Sum;
  bd.arg = bad.aggs[0]->args[0];
  EXPECT_THROW(gen_agg_plan(bad, {0}, true, {}, {}, {bd}), JitUnsupported);
  // arithmetic on strings
  PlannedQuery bad2 = e.plan("select count(*) from fact where flag + 1 > 2");
  EXPECT_THROW(gen_filter_plan(bad2, 0, bad2.push[0]), JitUnsupported);
}

}  // namespace
