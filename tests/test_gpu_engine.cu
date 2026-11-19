// GPU backend vs CPU engine parity on small tables (fast; no TPC-H data needed), plus allocator tests.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

#include "exec/engine.h"
#include "gpu/gpu_backend.h"
#include "gpu/memory.cuh"
#include "storage/csv.h"

using namespace warpdb;

namespace {

std::string write_tmp(const std::string& name, const std::string& body) {
  std::string path = (std::filesystem::temp_directory_path() / name).string();
  std::ofstream(path, std::ios::binary) << body;
  return path;
}

// Deterministic synthetic star schema: a fact table with ~20k rows, two dimensions, string + date + double columns.
void make_data(Engine& eng) {
  std::mt19937 g(1234);
  std::string fact = "id,cust,prod,qty,price,day,flag\n";
  const char* flags[] = {"A", "B", "C"};
  for (int i = 0; i < 20000; ++i) {
    int day = 18000 + static_cast<int>(g() % 700);
    int y;
    unsigned m, d;
    civil_from_days(day, y, m, d);
    char buf[160];
    std::snprintf(buf, sizeof buf, "%d,%d,%d,%d,%.2f,%04d-%02u-%02u,%s\n", i, static_cast<int>(g() % 500),
                  static_cast<int>(g() % 40), 1 + static_cast<int>(g() % 50), (g() % 100000) / 100.0, y, m, d,
                  flags[g() % 3]);
    fact += buf;
  }
  std::string cust = "cid,region,seg\n";
  const char* regions[] = {"ASIA", "EUROPE", "AMERICA", "AFRICA"};
  for (int c = 0; c < 500; ++c) cust += std::to_string(c) + "," + regions[c % 4] + ",seg" + std::to_string(c % 7) + "\n";
  std::string prod = "pid,cat,name\n";
  for (int p = 0; p < 40; ++p) prod += std::to_string(p) + ",cat" + std::to_string(p % 5) + ",item " + std::to_string(p) + (p % 3 ? " red" : " blue") + "\n";
  eng.catalog.add(load_csv(write_tmp("wdb_fact.csv", fact), "fact"));
  eng.catalog.add(load_csv(write_tmp("wdb_cust.csv", cust), "cust"));
  eng.catalog.add(load_csv(write_tmp("wdb_prod.csv", prod), "prod"));
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
  if (*ea || *eb) return false;  // not both numeric
  return std::fabs(x - y) <= 1e-9 * std::max(1.0, std::max(std::fabs(x), std::fabs(y)));
}

struct GpuParity : ::testing::Test {
  static Engine& eng() {
    static Engine e;
    static bool init = false;
    if (!init) {
      make_data(e);
      e.set_gpu_backend(gpu::make_gpu_backend());
      init = true;
    }
    return e;
  }
  void expect_same(const std::string& sql) {
    Engine& e = eng();
    e.device = "cpu";
    QueryResult cpu = e.execute(sql);
    e.device = "gpu";
    QueryResult gpu_r = e.execute(sql);
    e.device = "cpu";
    bool ordered = sql.find("order by") != std::string::npos;
    auto a = rows_of(cpu, !ordered), b = rows_of(gpu_r, !ordered);
    ASSERT_EQ(a.size(), b.size()) << sql;
    ASSERT_EQ(cpu.names, gpu_r.names) << sql;
    for (size_t i = 0; i < a.size(); ++i)
      for (size_t j = 0; j < a[i].size(); ++j)
        ASSERT_TRUE(cells_equal(a[i][j], b[i][j])) << sql << "\n row " << i << " col " << j << ": cpu=" << a[i][j]
                                                    << " gpu=" << b[i][j];
  }
};

TEST_F(GpuParity, FiltersAndExpressions) {
  expect_same("select count(*) from fact where qty > 25");
  expect_same("select count(*), sum(qty * price) from fact where qty between 10 and 20 and price < 500.5");
  expect_same("select count(*) from fact where flag = 'B' or (qty < 5 and not flag = 'A')");
  expect_same("select count(*) from fact where flag in ('A','C') and day >= date '2019-06-01' - interval '1' month");
  expect_same("select sum(case when qty > 25 then price else 0 end), sum(case when flag = 'A' then 1 else 0 end) from fact");
  expect_same("select count(*) from fact where -qty < -40 and abs(price - 500) < 10");
}

TEST_F(GpuParity, GroupByAndAggregates) {
  expect_same("select flag, count(*), sum(qty), avg(price), min(price), max(qty) from fact group by flag order by flag");
  expect_same("select cust, sum(price) as p, count(*) as c from fact group by cust order by p desc, cust limit 20");
  expect_same("select cust, prod, count(*) from fact group by cust, prod order by cust, prod limit 50");
  expect_same("select extract(year from day) as y, extract(month from day) as m, count(*) from fact group by y, m order by y, m");
  expect_same("select flag, max(day), min(day) from fact group by flag order by flag");
  expect_same("select prod from fact group by prod having count(*) > 520 order by prod");
  expect_same("select sum(price) / count(*), max(price) - min(price) from fact");
  expect_same("select count(*), sum(price), avg(qty) from fact where qty > 1000");  // empty input
}

TEST_F(GpuParity, JoinsOnIntAndString) {
  expect_same("select region, count(*), sum(price) from fact, cust where cust = cid group by region order by region");
  expect_same("select cat, count(*) from fact join prod on prod = pid where qty > 10 group by cat order by cat");
  expect_same("select region, cat, sum(qty) from fact, cust, prod where cust = cid and prod = pid and flag = 'A' "
              "group by region, cat order by region, cat");
  expect_same("select count(*) from cust c1, cust c2 where c1.seg = c2.seg and c1.cid < c2.cid");  // VARCHAR key + residual
  expect_same("select name from prod where name like '%red' order by name");
  expect_same("select count(*) from fact, prod where prod = pid and name like 'item 1%'");
}

TEST_F(GpuParity, SortLimitDistinct) {
  expect_same("select id, price from fact order by price desc, id limit 25");
  expect_same("select id from fact where qty = 7 order by day, id desc limit 10");
  expect_same("select distinct flag, day from fact where qty < 3 order by day, flag limit 30");
  expect_same("select id from fact order by flag, qty desc, id limit 100");
  expect_same("select * from prod order by pid limit 5");
}

TEST_F(GpuParity, ColumnsStayResidentAcrossQueries) {
  Engine& e = eng();
  e.device = "gpu";
  e.execute("select sum(price) from fact");
  QueryResult warm = e.execute("select sum(price) from fact");
  e.device = "cpu";
  bool saw_upload = false;
  for (auto& t : warm.trace) saw_upload |= t.find("column upload") != std::string::npos;
  EXPECT_FALSE(saw_upload) << "second query should hit the VRAM column cache";
}

TEST(GpuAllocator, ReusesFreedBlocksAndTracksUsage) {
  auto& a = gpu::GpuAllocator::instance();
  size_t base_use = a.in_use();
  void* p = a.alloc(10 << 20);
  size_t reserved1 = a.reserved();
  EXPECT_GT(a.in_use(), base_use);
  a.free(p);
  EXPECT_EQ(a.in_use(), base_use);
  void* q = a.alloc(10 << 20);  // same size class: must come from the cache, not cudaMalloc
  EXPECT_EQ(q, p);
  EXPECT_EQ(a.reserved(), reserved1);
  a.free(q);
}

TEST(GpuAllocator, ThrowsReadableErrorWhenOutOfMemory) {
  EXPECT_THROW(
      {
        try {
          gpu::GpuVec<uint8_t> huge(size_t(1) << 44);  // 16 TB
        } catch (const std::runtime_error& e) {
          EXPECT_NE(std::string(e.what()).find("out of memory"), std::string::npos);
          throw;
        }
      },
      std::runtime_error);
}

}  // namespace
