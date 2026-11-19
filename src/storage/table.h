#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/util.h"

namespace warpdb {

enum class Type : uint8_t { INT32, INT64, FLOAT64, DATE, VARCHAR };

inline const char* type_name(Type t) {
  switch (t) {
    case Type::INT32: return "INT32";
    case Type::INT64: return "INT64";
    case Type::FLOAT64: return "FLOAT64";
    case Type::DATE: return "DATE";
    case Type::VARCHAR: return "VARCHAR";
  }
  return "?";
}

// A fixed-width columnar vector. Exactly one of the payload vectors is used:
//   INT32 / DATE / VARCHAR(codes) -> i32,  INT64 -> i64,  FLOAT64 -> f64.
// VARCHAR is dictionary-encoded and the dictionary is SORTED, so code order == string order
// (range predicates, ORDER BY and GROUP BY can all work on the 4-byte codes).
// NULLs are not supported in v1 (SSB / TPC-H contain none).
struct Column {
  Type type = Type::INT32;
  std::vector<int32_t> i32;
  std::vector<int64_t> i64;
  std::vector<double> f64;
  std::vector<std::string> dict;

  size_t size() const {
    switch (type) {
      case Type::INT64: return i64.size();
      case Type::FLOAT64: return f64.size();
      default: return i32.size();
    }
  }
  size_t bytes() const {
    size_t b = i32.size() * 4 + i64.size() * 8 + f64.size() * 8;
    for (auto& s : dict) b += s.size() + 32;
    return b;
  }
};

struct Table {
  std::string name;
  std::vector<std::string> names;  // lowercase
  std::vector<std::shared_ptr<Column>> cols;
  size_t nrows = 0;

  int find(const std::string& lname) const {
    for (size_t i = 0; i < names.size(); ++i)
      if (names[i] == lname) return static_cast<int>(i);
    return -1;
  }
  size_t bytes() const {
    size_t b = 0;
    for (auto& c : cols) b += c->bytes();
    return b;
  }
};

class Catalog {
 public:
  void add(std::shared_ptr<Table> t) { tables_[to_lower(t->name)] = std::move(t); }
  std::shared_ptr<Table> get(const std::string& name) const {
    auto it = tables_.find(to_lower(name));
    return it == tables_.end() ? nullptr : it->second;
  }
  const std::unordered_map<std::string, std::shared_ptr<Table>>& all() const { return tables_; }

 private:
  std::unordered_map<std::string, std::shared_ptr<Table>> tables_;
};

}  // namespace warpdb
