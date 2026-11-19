#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "exec/engine.h"
#include "storage/csv.h"

using namespace warpdb;

namespace {

std::string write_tmp(const std::string& name, const std::string& body) {
  std::string path = (std::filesystem::temp_directory_path() / name).string();
  std::ofstream(path, std::ios::binary) << body;
  return path;
}

struct Fixture : ::testing::Test {
  Engine eng;
  void SetUp() override {
    eng.catalog.add(load_csv(write_tmp("wdb_emp.csv",
                                       "id,dept,salary,hired,name\n"
                                       "1,eng,100.5,2020-01-15,ann\n"
                                       "2,eng,80,2019-06-01,bob\n"
                                       "3,ops,60,2021-03-20,\"cy, jr\"\n"
                                       "4,ops,40,2018-11-30,dee\n"
                                       "5,hr,55,2022-07-04,eve\n"),
                             "emp"));
    eng.catalog.add(load_csv(write_tmp("wdb_dept.csv", "dept,floor\neng,3\nops,1\nhr,2\n"), "dept"));
  }
  // first column of every row, as strings
  std::vector<std::string> col(const std::string& sql, size_t c = 0) {
    auto r = eng.execute(sql);
    std::vector<std::string> v;
    for (size_t i = 0; i < r.nrows; ++i) v.push_back(format_cell(*r.cols[c], i));
    return v;
  }
};

using V = std::vector<std::string>;

TEST_F(Fixture, TypeInference) {
  auto t = eng.catalog.get("emp");
  EXPECT_EQ(t->cols[0]->type, Type::INT32);
  EXPECT_EQ(t->cols[2]->type, Type::FLOAT64);
  EXPECT_EQ(t->cols[3]->type, Type::DATE);
  EXPECT_EQ(t->cols[4]->type, Type::VARCHAR);
  EXPECT_EQ(t->cols[4]->dict[0], "ann");  // sorted dictionary
  EXPECT_EQ(t->cols[4]->dict[2], "cy, jr");  // quoted field with a comma
}

TEST_F(Fixture, FilterOrderLimit) {
  EXPECT_EQ(col("select id from emp where salary > 50 order by salary desc limit 3"), (V{"1", "2", "3"}));
  EXPECT_EQ(col("select name from emp where hired between date '2019-01-01' and date '2021-12-31' order by id"),
            (V{"ann", "bob", "cy, jr"}));
}

TEST_F(Fixture, GroupByAggregates) {
  auto r = eng.execute("select dept, count(*) as c, sum(salary) as s, avg(salary) as a, min(id) as lo, max(name) as mx "
                       "from emp group by dept order by dept");
  ASSERT_EQ(r.nrows, 3u);
  EXPECT_EQ(format_cell(*r.cols[0], 0), "eng");
  EXPECT_EQ(format_cell(*r.cols[1], 0), "2");
  EXPECT_EQ(format_cell(*r.cols[2], 0), "180.5");
  EXPECT_EQ(format_cell(*r.cols[0], 1), "hr");
  EXPECT_EQ(format_cell(*r.cols[4], 1), "5");
  EXPECT_EQ(format_cell(*r.cols[5], 2), "dee");  // max over a VARCHAR column
}

TEST_F(Fixture, HavingAndAggregateArithmetic) {
  EXPECT_EQ(col("select dept from emp group by dept having sum(salary) >= 100 order by dept"), (V{"eng", "ops"}));
  EXPECT_EQ(col("select dept from emp group by dept having sum(salary) > 100 order by dept"), (V{"eng"}));
  EXPECT_EQ(col("select sum(salary) / count(*) from emp"), (V{"67.1"}));
}

TEST_F(Fixture, GlobalAggregateOnEmptyInputYieldsOneRow) {
  EXPECT_EQ(col("select count(*) from emp where salary > 1000"), (V{"0"}));
}

TEST_F(Fixture, JoinOnAndCommaSyntax) {
  EXPECT_EQ(col("select count(*) from emp join dept on emp.dept = dept.dept where floor >= 2"), (V{"3"}));
  EXPECT_EQ(col("select sum(floor) from emp e, dept d where e.dept = d.dept"), (V{"10"}));  // 3*2 + 1*2 + 2*1
}

TEST_F(Fixture, LikeInDistinctExtract) {
  EXPECT_EQ(col("select count(*) from emp where name like '%e%'"), (V{"2"}));
  EXPECT_EQ(col("select id from emp where dept in ('hr','ops') order by id"), (V{"3", "4", "5"}));
  EXPECT_EQ(col("select distinct dept from emp order by dept"), (V{"eng", "hr", "ops"}));
  EXPECT_EQ(col("select extract(year from hired) as y from emp order by y limit 1"), (V{"2018"}));
}

TEST_F(Fixture, DateIntervalFolding) {
  // 2020-01-15 + 1 month = 2020-02-15  -> hired before it: 2020-01-15, 2019-06-01, 2018-11-30
  EXPECT_EQ(col("select count(*) from emp where hired < date '2020-01-15' + interval '1' month"), (V{"3"}));
  // 2022-01-01 - 1 year = 2021-01-01 -> 2021-03-20, 2022-07-04
  EXPECT_EQ(col("select count(*) from emp where hired >= date '2022-01-01' - interval '1' year"), (V{"2"}));
}

TEST_F(Fixture, Errors) {
  EXPECT_THROW(eng.execute("select nope from emp"), Error);
  EXPECT_THROW(eng.execute("select * from missing"), Error);
  EXPECT_THROW(eng.execute("select id from emp where"), Error);
  EXPECT_THROW(eng.execute("select dept, salary from emp group by dept"), Error);
  EXPECT_THROW(eng.execute("select id from emp where id in (select id from emp)"), Error);
  EXPECT_THROW(eng.execute("select * from emp, dept"), Error);  // cross join unsupported
}

TEST(Dates, RoundTrip) {
  int32_t d;
  ASSERT_TRUE(parse_date("1970-01-01", d));
  EXPECT_EQ(d, 0);
  ASSERT_TRUE(parse_date("2000-02-29", d));
  EXPECT_EQ(format_date(d), "2000-02-29");
  EXPECT_FALSE(parse_date("2001-02-29", d));
}

}  // namespace
