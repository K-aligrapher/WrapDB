#pragma once
#include <memory>
#include <string>
#include <vector>

#include "storage/table.h"

namespace warpdb {

struct CsvOptions {
  char delimiter = ',';
  bool header = true;
  std::vector<std::string> column_names;  // used when header == false
};

// Loads a delimited text file into a columnar table. Types are inferred over the whole file
// (INT32 < INT64 < FLOAT64, DATE for yyyy-mm-dd, otherwise VARCHAR). Quoted fields are supported
// (without embedded quotes/newlines). A trailing delimiter on every line (TPC-H .tbl) is tolerated.
std::shared_ptr<Table> load_csv(const std::string& path, const std::string& table_name,
                                const CsvOptions& opt = {});

}  // namespace warpdb
