// Placement optimizer: cost-model fitting, persistence, estimator accuracy and decision sanity.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "exec/engine.h"
#include "gpu/gpu_backend.h"
#include "optimizer/cost_model.h"

using namespace warpdb;

// ------------------------------------------------------------------------------------------------ cost model
TEST(LinearModel, RecoversInterceptAndSlopeFromNoisyData) {
  LinearModel m;
  std::mt19937 g(1);
  std::normal_distribution<double> noise(0.0, 0.01);
  for (int i = 0; i < 400; ++i) {
    double x = std::pow(10.0, 3.0 + 4.0 * (g() % 1000) / 1000.0);  // 1e3 .. 1e7, log-uniform
    double y = (0.5 + 2e-6 * x) * (1.0 + noise(g));
    m.observe(x, y);
  }
  EXPECT_NEAR(m.a(), 0.5, 0.05);
  EXPECT_NEAR(m.b(), 2e-6, 2e-7);
  EXPECT_NEAR(m.predict(1e6), 2.5, 0.15);
}

TEST(LinearModel, RelativeWeightingKeepsSmallQueriesAccurate) {
  // Heavy-tailed noise on large inputs must not wreck the fit for tiny ones.
  LinearModel m;
  std::mt19937 g(2);
  for (int i = 0; i < 300; ++i) {
    double x = (i % 2) ? 1e3 : 1e7;
    double y = 0.1 + 1e-6 * x;
    if (x > 1e6 && g() % 5 == 0) y *= 1.5;  // outliers on the big ones only
    m.observe(x, y);
  }
  EXPECT_NEAR(m.predict(1e3), 0.101, 0.03);
}

TEST(LinearModel, ParametersStayPhysical) {
  LinearModel m;
  for (int i = 0; i < 50; ++i) m.observe(100.0 + i, 5.0 - 0.01 * i);  // decreasing: nonsense data
  EXPECT_GE(m.a(), 0.0);
  EXPECT_GE(m.b(), 0.0);
  EXPECT_GE(m.predict(1e9), 0.0);
}

TEST(LinearModel, TracksDrift) {
  LinearModel m;
  for (int i = 0; i < 200; ++i) m.observe(1e6 + 1000.0 * (i % 7) * 1e3, 1.0 + 1e-6 * (1e6 + 1000.0 * (i % 7) * 1e3));
  double before = m.predict(5e6);
  for (int i = 0; i < 1500; ++i) {  // the device became 3x slower (thermals, other load)
    double x = 1e6 + 1e6 * (i % 7);
    m.observe(x, 3.0 * (1.0 + 1e-6 * x));
  }
  EXPECT_GT(m.predict(5e6), 2.0 * before);
}

TEST(StageWork, ScalesWithTheDataTheStageTouches) {
  StageStat s;
  s.op = StageOp::Scan;
  s.rows_in = 1e6;
  s.k1 = 1;
  double w1 = stage_work(s);
  s.k1 = 3;
  EXPECT_GT(stage_work(s), 2.0 * w1);   // more columns -> more work
  s.k2 = 5;
  EXPECT_GT(stage_work(s), w1 * 3.0);   // more operator nodes -> more work
  StageStat j;
  j.op = StageOp::Join;
  j.rows_in = 1e6; j.rows_aux = 1e5; j.rows_out = 1e6; j.k1 = 1; j.k2 = 2;
  double wj = stage_work(j);
  j.rows_out = 1e7;
  EXPECT_GT(stage_work(j), wj);
}

TEST(CostModel, JsonRoundTripAndRejectsForeignHardware) {
  CostModel a;
  a.hw.fingerprint = "GPU-X|threads=8";
  a.hw.has_gpu = true;
  a.hw.h2d_gbs = 12.5;
  a.calibrated = true;
  for (int i = 0; i < 20; ++i) a.model(StageOp::Join, true).observe(1e5 * (i + 1), 0.2 + 3e-7 * 1e5 * (i + 1));
  CostModel b;
  ASSERT_TRUE(b.from_json(a.to_json(), "GPU-X|threads=8"));
  EXPECT_TRUE(b.calibrated);
  EXPECT_NEAR(b.hw.h2d_gbs, 12.5, 1e-6);
  EXPECT_NEAR(b.model(StageOp::Join, true).a(), a.model(StageOp::Join, true).a(), 1e-9);
  EXPECT_NEAR(b.model(StageOp::Join, true).b(), a.model(StageOp::Join, true).b(), 1e-12);
  EXPECT_EQ(b.model(StageOp::Join, true).samples(), 20u);
  CostModel c;
  EXPECT_FALSE(c.from_json(a.to_json(), "OTHER-GPU|threads=8"));  // a profile of a different machine is ignored
  EXPECT_FALSE(c.from_json("{\"version\": 1}", "GPU-X|threads=8"));  // stale format is ignored
}

// ------------------------------------------------------------------------------------------------ estimator + decisions
namespace {

std::shared_ptr<Table> make_table(const std::string& name, size_t n, uint32_t seed) {
  std::mt19937 g(seed);
  auto t = std::make_shared<Table>();
  t->name = name;
  t->nrows = n;
  t->names = {"id", "g", "x", "y"};
  auto c_id = std::make_shared<Column>(), c_g = std::make_shared<Column>(), c_x = std::make_shared<Column>(),
       c_y = std::make_shared<Column>();
  c_id->type = c_g->type = Type::INT32;
  c_x->type = c_y->type = Type::FLOAT64;
  for (size_t i = 0; i < n; ++i) {
    c_id->i32.push_back(static_cast<int32_t>(i));
    c_g->i32.push_back(static_cast<int32_t>(g() % 20));
    c_x->f64.push_back((g() % 1000000) / 1e6);  // uniform [0,1)
    c_y->f64.push_back((g() % 100000) / 1e3);
  }
  t->cols = {c_id, c_g, c_x, c_y};
  return t;
}

struct Optimizer : ::testing::Test {
  static Engine& eng() {
    static Engine e;
    static bool init = false;
    if (!init) {
      e.set_gpu_backend(gpu::make_gpu_backend());
      e.catalog.add(make_table("big", 3000000, 7));
      e.catalog.add(make_table("tiny", 40, 8));
      e.calibrate();  // ~3 s, once for the whole suite; no profile file is written (profile_path is empty)
      init = true;
    }
    return e;
  }
};

TEST_F(Optimizer, SampledSelectivityIsCloseToTruth) {
  Engine& e = eng();
  PlannedQuery q = e.plan("select count(*) from big where x < 0.25");
  EXPECT_NEAR(e.sample_selectivity(q, 0), 0.25, 0.04);
  PlannedQuery q2 = e.plan("select count(*) from big where x < 0.5 and g = 3");
  EXPECT_NEAR(e.sample_selectivity(q2, 0), 0.5 * 0.05, 0.02);
  PlannedQuery q3 = e.plan("select count(*) from big");
  EXPECT_DOUBLE_EQ(e.sample_selectivity(q3, 0), 1.0);
}

TEST_F(Optimizer, ExpressionKeyCardinalityComesFromTheData) {
  Engine& e = eng();
  PlannedQuery q = e.plan("select g % 7, count(*) from big group by g % 7");
  double ndv = e.key_ndv(q, q.sel.group_by[0], 3000000);
  EXPECT_LT(ndv, 20.0);  // 7 distinct values, not 3 million
  EXPECT_GE(ndv, 5.0);
  PlannedQuery u = e.plan("select id + 1, count(*) from big group by id + 1");
  EXPECT_GT(e.key_ndv(u, u.sel.group_by[0], 3000000), 1e6);  // (nearly) unique expression
}

TEST_F(Optimizer, TinyQueriesStayOnTheCpuAndBigWarmOnesGoToTheGpu) {
  Engine& e = eng();
  PlannedQuery tiny = e.plan("select g, count(*), sum(x) from tiny group by g");
  PlacementEstimate te = e.estimate_placement(tiny);
  EXPECT_EQ(te.choice, "cpu") << "cpu " << te.cpu_ms << " ms vs gpu " << te.gpu_ms << " ms";

  // make the big table's columns resident, then a scan/aggregate over it must be estimated cheaper on the GPU
  e.device = "gpu";
  e.execute("select g, sum(x), avg(y) from big where x < 0.9 group by g");
  e.device = "cpu";
  PlannedQuery big = e.plan("select g, sum(x), avg(y) from big where x < 0.9 group by g");
  PlacementEstimate be = e.estimate_placement(big);
  EXPECT_EQ(be.cols_cold, 0);
  EXPECT_EQ(be.choice, "gpu") << "cpu " << be.cpu_ms << " ms vs gpu " << be.gpu_ms << " ms";
  EXPECT_GT(be.cpu_ms, 3.0 * be.gpu_ms);
}

TEST_F(Optimizer, ColdColumnsAddUploadCostAndWarmColumnsDoNot) {
  Engine& e = eng();
  e.gpu_backend()->drop_cache();
  PlannedQuery q = e.plan("select sum(x * y) from big where g < 10");
  PlacementEstimate cold = e.estimate_placement(q);
  EXPECT_EQ(cold.cols_cold, cold.cols_total);
  EXPECT_GT(cold.upload_mb, 50.0);  // 3 columns x 3M rows (4 + 8 + 8 bytes)
  EXPECT_GT(cold.upload_ms, 1.0);
  e.device = "gpu";
  e.execute("select sum(x * y) from big where g < 10");
  e.device = "cpu";
  PlacementEstimate warm = e.estimate_placement(q);
  EXPECT_EQ(warm.cols_cold, 0);
  EXPECT_DOUBLE_EQ(warm.upload_ms, 0.0);
  EXPECT_LT(warm.gpu_ms, cold.gpu_ms);
}

TEST_F(Optimizer, ExplainEstimatesWithoutExecutingAndAnalyzeExecutes) {
  Engine& e = eng();
  e.device = "auto";
  QueryResult plan_only = e.execute("explain select g, count(*) from big group by g");
  EXPECT_TRUE(plan_only.explain);
  EXPECT_EQ(plan_only.nrows, 0u);  // nothing was executed
  EXPECT_TRUE(plan_only.stages.empty());
  bool has_decision = false;
  for (auto& l : plan_only.plan_lines) has_decision |= l.find("decision:") != std::string::npos;
  EXPECT_TRUE(has_decision);

  QueryResult analyzed = e.execute("explain analyze select g, count(*) from big group by g");
  EXPECT_TRUE(analyzed.explain);
  EXPECT_FALSE(analyzed.stages.empty());  // it ran
  EXPECT_FALSE(analyzed.trace.empty());
  e.device = "cpu";
}

TEST_F(Optimizer, AutoModeRecordsDecisionsAndLearnsFromExecutions) {
  Engine& e = eng();
  e.device = "auto";
  size_t before = e.decisions.size();
  size_t samples_before = e.cost.model(StageOp::Scan, false).samples() + e.cost.model(StageOp::Scan, true).samples();
  QueryResult r = e.execute("select count(*) from big where x < 0.1");
  EXPECT_EQ(r.nrows, 1u);
  EXPECT_FALSE(r.device_used.empty());
  ASSERT_EQ(e.decisions.size(), before + 1);
  const DecisionRecord& d = e.decisions.back();
  EXPECT_EQ(d.chosen, r.device_used);
  EXPECT_GT(d.est_cpu_ms, 0.0);
  EXPECT_LT(d.decide_ms, 50.0);  // deciding is cheap relative to executing
  size_t samples_after = e.cost.model(StageOp::Scan, false).samples() + e.cost.model(StageOp::Scan, true).samples();
  EXPECT_GT(samples_after, samples_before);
  e.device = "cpu";
}

TEST_F(Optimizer, AutoAndForcedDevicesReturnTheSameAnswer) {
  Engine& e = eng();
  const char* sql = "select g, count(*), sum(x) from big where y > 50 group by g order by g";
  e.device = "cpu";
  QueryResult c = e.execute(sql);
  e.device = "auto";
  QueryResult a = e.execute(sql);
  e.device = "cpu";
  ASSERT_EQ(c.nrows, a.nrows);
  for (size_t i = 0; i < c.nrows; ++i) {
    EXPECT_EQ(format_cell(*c.cols[0], i), format_cell(*a.cols[0], i));
    EXPECT_EQ(format_cell(*c.cols[1], i), format_cell(*a.cols[1], i));
    EXPECT_NEAR(std::stod(format_cell(*c.cols[2], i)), std::stod(format_cell(*a.cols[2], i)), 1e-6 * 1e5);
  }
}

}  // namespace
