// Python bindings (pybind11): the engine as a library. Results come back as NumPy arrays (dates as day counts,
// VARCHAR as compact category codes) so pandas conversion is cheap; optimizer / GPU / JIT introspection is exposed
// as plain dicts and lists for the dashboard.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <mutex>

#include "exec/engine.h"
#include "gpu/device.h"
#include "gpu/gpu_backend.h"
#include "storage/csv.h"

namespace py = pybind11;
using namespace warpdb;

namespace {

py::dict stage_to_dict(const StageStat& s) {
  py::dict d;
  d["op"] = stage_op_name(s.op);
  d["device"] = s.gpu ? "gpu" : "cpu";
  d["label"] = s.label;
  d["rows_in"] = s.rows_in;
  d["rows_aux"] = s.rows_aux;
  d["rows_out"] = s.rows_out;
  d["k1"] = s.k1;
  d["k2"] = s.k2;
  d["k3"] = s.k3;
  d["ms"] = s.ms;
  d["work"] = stage_work(s);
  return d;
}

py::dict estimate_to_dict(const PlacementEstimate& e) {
  py::dict d;
  py::list stages;
  for (auto& se : e.stages) {
    py::dict s = stage_to_dict(se.stat);
    s["cpu_ms"] = se.cpu_ms;
    s["gpu_ms"] = se.gpu_ms;
    stages.append(s);
  }
  d["stages"] = stages;
  d["cols_total"] = e.cols_total;
  d["cols_cold"] = e.cols_cold;
  d["upload_mb"] = e.upload_mb;
  d["upload_ms"] = e.upload_ms;
  d["cpu_ms"] = e.cpu_ms;
  d["gpu_ms"] = e.gpu_ms;
  d["gpu_feasible"] = e.gpu_feasible;
  d["gpu_reason"] = e.gpu_reason;
  d["choice"] = e.choice;
  d["lines"] = e.lines;
  return d;
}

class PyResult {
 public:
  explicit PyResult(QueryResult r) : r_(std::move(r)) {}
  size_t nrows() const { return r_.nrows; }
  double ms() const { return r_.ms; }
  const std::string& device() const { return r_.device_used; }
  const std::vector<std::string>& names() const { return r_.names; }
  const std::vector<std::string>& trace() const { return r_.trace; }
  const std::vector<std::string>& plan_lines() const { return r_.plan_lines; }
  bool is_explain() const { return r_.explain; }
  py::list stages() const {
    py::list l;
    for (auto& s : r_.stages) l.append(stage_to_dict(s));
    return l;
  }

  // One dict per column: {name, kind: int|float|bool|date|str, data: ndarray, categories: list[str] (str only)}
  py::list columns() const {
    py::list out;
    for (size_t j = 0; j < r_.cols.size(); ++j) {
      const Value& v = *r_.cols[j];
      py::dict c;
      c["name"] = r_.names[j];
      const size_t n = r_.nrows;
      switch (v.kind) {
        case Value::I: {
          c["kind"] = v.is_date ? "date" : "int";
          py::array_t<int64_t> a(n);
          if (n) std::memcpy(a.mutable_data(), v.i.data(), n * 8);
          c["data"] = a;
          break;
        }
        case Value::F: {
          c["kind"] = "float";
          py::array_t<double> a(n);
          if (n) std::memcpy(a.mutable_data(), v.f.data(), n * 8);
          c["data"] = a;
          break;
        }
        case Value::B: {
          c["kind"] = "bool";
          py::array_t<bool> a(n);
          for (size_t i = 0; i < n; ++i) a.mutable_data()[i] = v.b[i] != 0;
          c["data"] = a;
          break;
        }
        case Value::S: {  // compact categories: only the dictionary entries actually used by this result
          c["kind"] = "str";
          const auto& dict = v.dict->dict;
          std::vector<int32_t> remap(dict.size(), -1);
          std::vector<std::string> cats;
          py::array_t<int32_t> codes(n);
          for (size_t i = 0; i < n; ++i) {
            size_t code = static_cast<size_t>(v.i[i]);
            if (remap[code] < 0) {
              remap[code] = static_cast<int32_t>(cats.size());
              cats.push_back(dict[code]);
            }
            codes.mutable_data()[i] = remap[code];
          }
          c["data"] = codes;
          c["categories"] = cats;
          break;
        }
        default: throw std::runtime_error("unsupported result column type");
      }
      out.append(c);
    }
    return out;
  }

 private:
  QueryResult r_;
};

class PyDatabase {
 public:
  explicit PyDatabase(bool use_gpu) {
    if (use_gpu) {
      gpu::DeviceInfo d;
      if (gpu::query_device(d)) {
        try {
          gpu_ = gpu::make_gpu_backend();
          eng_.set_gpu_backend(gpu_);
          gpu_name_ = d.name;
        } catch (const std::exception&) {
          gpu_.reset();
        }
      }
    }
    const char* appdata = std::getenv("LOCALAPPDATA");
    eng_.profile_path = appdata ? std::string(appdata) + "\\warpdb\\profile.json" : ".warpdb_profile.json";
    eng_.device = gpu_ ? "auto" : "cpu";
  }

  py::dict load_csv(const std::string& name, const std::string& path, const std::string& delimiter, bool header) {
    std::lock_guard<std::mutex> g(mu_);
    CsvOptions o;
    o.header = header;
    if (!delimiter.empty()) o.delimiter = delimiter[0];
    Timer t;
    std::shared_ptr<Table> tab;
    {
      py::gil_scoped_release release;
      tab = warpdb::load_csv(path, name, o);
    }
    eng_.catalog.add(tab);
    py::dict d;
    d["name"] = tab->name;
    d["rows"] = tab->nrows;
    d["columns"] = tab->names.size();
    d["mb"] = tab->bytes() / 1048576.0;
    d["ms"] = t.ms();
    return d;
  }

  py::list tables() {
    std::lock_guard<std::mutex> g(mu_);
    py::list out;
    for (auto& kv : eng_.catalog.all()) {
      const Table& t = *kv.second;
      py::dict d;
      d["name"] = t.name;
      d["rows"] = t.nrows;
      d["mb"] = t.bytes() / 1048576.0;
      py::list cols;
      for (size_t i = 0; i < t.names.size(); ++i) {
        py::dict c;
        c["name"] = t.names[i];
        c["type"] = type_name(t.cols[i]->type);
        cols.append(c);
      }
      d["columns"] = cols;
      out.append(d);
    }
    return out;
  }

  PyResult execute(const std::string& sql, const std::string& device) {
    std::lock_guard<std::mutex> g(mu_);
    std::string saved = eng_.device;
    if (!device.empty()) eng_.device = device;
    QueryResult r;
    try {
      py::gil_scoped_release release;
      r = eng_.execute(sql);
    } catch (...) {
      eng_.device = saved;
      throw;
    }
    eng_.device = saved;
    return PyResult(std::move(r));
  }

  py::dict estimate(const std::string& sql) {
    std::lock_guard<std::mutex> g(mu_);
    PlacementEstimate e;
    {
      py::gil_scoped_release release;
      eng_.ensure_calibrated();
      PlannedQuery pq = eng_.plan(sql);
      e = eng_.estimate_placement(pq);
    }
    return estimate_to_dict(e);
  }

  std::string get_device() const { return eng_.device; }
  void set_device(const std::string& d) {
    if (d != "cpu" && d != "gpu" && d != "auto") throw std::invalid_argument("device must be cpu, gpu or auto");
    if (d != "cpu" && !gpu_) throw std::runtime_error("no CUDA device available");
    eng_.device = d;
  }
  bool has_gpu() const { return gpu_ != nullptr; }
  std::string gpu_name() const { return gpu_name_; }
  bool get_learn() const { return eng_.learn; }
  void set_learn(bool b) { eng_.learn = b; }

  void calibrate() {
    std::lock_guard<std::mutex> g(mu_);
    py::gil_scoped_release release;
    eng_.calibrate();
  }
  void ensure_calibrated() {
    std::lock_guard<std::mutex> g(mu_);
    py::gil_scoped_release release;
    eng_.ensure_calibrated();
  }
  bool calibrated() const { return eng_.cost.calibrated; }

  py::dict cost_model() {
    std::lock_guard<std::mutex> g(mu_);
    py::dict d;
    const auto& hw = eng_.cost.hw;
    py::dict h;
    h["fingerprint"] = hw.fingerprint;
    h["has_gpu"] = hw.has_gpu;
    h["h2d_gbs"] = hw.h2d_gbs;
    h["d2h_gbs"] = hw.d2h_gbs;
    h["launch_us"] = hw.launch_us;
    h["gpu_membw_gbs"] = hw.gpu_membw_gbs;
    d["hardware"] = h;
    py::list models;
    for (int op = 0; op < kNumStageOps; ++op)
      for (int g2 = 0; g2 < 2; ++g2) {
        const auto& m = eng_.cost.model(static_cast<StageOp>(op), g2 != 0);
        py::dict e;
        e["stage"] = stage_op_name(static_cast<StageOp>(op));
        e["device"] = g2 ? "gpu" : "cpu";
        e["a_ms"] = m.a();
        e["b_ms_per_unit"] = m.b();
        e["samples"] = m.samples();
        models.append(e);
      }
    d["models"] = models;
    d["calibrated"] = eng_.cost.calibrated;
    return d;
  }

  py::list decisions() {
    std::lock_guard<std::mutex> g(mu_);
    py::list out;
    for (auto& r : eng_.decisions) {
      py::dict d;
      d["sql"] = r.sql;
      d["est_cpu_ms"] = r.est_cpu_ms;
      d["est_gpu_ms"] = r.est_gpu_ms;
      d["chosen"] = r.chosen;
      d["actual_ms"] = r.actual_ms;
      d["decide_ms"] = r.decide_ms;
      out.append(d);
    }
    return out;
  }

  py::dict gpu_stats() {
    std::lock_guard<std::mutex> g(mu_);
    py::dict d;
    if (!gpu_) return d;
    auto m = gpu_->mem_stats();
    d["cache_mb"] = m.cache_bytes / 1048576.0;
    d["cache_budget_mb"] = m.cache_budget / 1048576.0;
    d["in_use_mb"] = m.in_use / 1048576.0;
    d["peak_mb"] = m.peak / 1048576.0;
    d["reserved_mb"] = m.reserved / 1048576.0;
    d["uploaded_total_mb"] = m.uploaded_total / 1048576.0;
    return d;
  }
  void drop_cache() {
    std::lock_guard<std::mutex> g(mu_);
    if (gpu_) gpu_->drop_cache();
  }
  py::list residency() {
    std::lock_guard<std::mutex> g(mu_);
    py::list out;
    for (auto& kv : eng_.catalog.all()) {
      const Table& t = *kv.second;
      for (size_t i = 0; i < t.names.size(); ++i) {
        const Column& c = *t.cols[i];
        py::dict d;
        d["table"] = t.name;
        d["column"] = t.names[i];
        d["type"] = type_name(c.type);
        d["mb"] = c.size() * ((c.type == Type::INT64 || c.type == Type::FLOAT64) ? 8.0 : 4.0) / 1048576.0;
        d["resident"] = gpu_ ? gpu_->column_resident(c) : false;
        out.append(d);
      }
    }
    return out;
  }

  void set_jit(bool on) { if (gpu_) gpu_->set_jit(on); }
  bool jit_enabled() const { return gpu_ && gpu_->jit_enabled(); }
  void set_jit_min_rows(size_t n) { if (gpu_) gpu_->set_jit_min_rows(n); }
  size_t jit_preload(size_t max_kernels) {
    if (!gpu_) return 0;
    py::gil_scoped_release release;
    return gpu_->jit_preload(max_kernels);
  }
  py::dict jit_stats() const {
    py::dict d;
    if (!gpu_) return d;
    auto s = gpu_->jit_stats();
    d["compiles"] = s.compiles;
    d["mem_hits"] = s.mem_hits;
    d["disk_hits"] = s.disk_hits;
    d["failures"] = s.failures;
    d["compile_ms"] = s.compile_ms;
    d["load_ms"] = s.load_ms;
    return d;
  }

 private:
  Engine eng_;
  std::shared_ptr<gpu::GpuBackend> gpu_;
  std::string gpu_name_;
  std::mutex mu_;
};

}  // namespace

PYBIND11_MODULE(_warpdb, m) {
  m.doc() = "WarpDB: GPU-accelerated SQL engine";
  py::register_exception<Error>(m, "WarpError");

  py::class_<PyResult>(m, "Result")
      .def_property_readonly("nrows", &PyResult::nrows)
      .def_property_readonly("ms", &PyResult::ms)
      .def_property_readonly("device", &PyResult::device)
      .def_property_readonly("names", &PyResult::names)
      .def_property_readonly("trace", &PyResult::trace)
      .def_property_readonly("plan_lines", &PyResult::plan_lines)
      .def_property_readonly("is_explain", &PyResult::is_explain)
      .def("stages", &PyResult::stages)
      .def("columns", &PyResult::columns);

  py::class_<PyDatabase>(m, "Database")
      .def(py::init<bool>(), py::arg("use_gpu") = true)
      .def("load_csv", &PyDatabase::load_csv, py::arg("name"), py::arg("path"), py::arg("delimiter") = ",",
           py::arg("header") = true)
      .def("tables", &PyDatabase::tables)
      .def("execute", &PyDatabase::execute, py::arg("sql"), py::arg("device") = "")
      .def("estimate", &PyDatabase::estimate)
      .def_property("device", &PyDatabase::get_device, &PyDatabase::set_device)
      .def_property_readonly("has_gpu", &PyDatabase::has_gpu)
      .def_property_readonly("gpu_name", &PyDatabase::gpu_name)
      .def_property("learn", &PyDatabase::get_learn, &PyDatabase::set_learn)
      .def("calibrate", &PyDatabase::calibrate)
      .def("ensure_calibrated", &PyDatabase::ensure_calibrated)
      .def_property_readonly("calibrated", &PyDatabase::calibrated)
      .def("cost_model", &PyDatabase::cost_model)
      .def("decisions", &PyDatabase::decisions)
      .def("gpu_stats", &PyDatabase::gpu_stats)
      .def("drop_cache", &PyDatabase::drop_cache)
      .def("residency", &PyDatabase::residency)
      .def("set_jit", &PyDatabase::set_jit)
      .def_property_readonly("jit_enabled", &PyDatabase::jit_enabled)
      .def("set_jit_min_rows", &PyDatabase::set_jit_min_rows)
      .def("jit_stats", &PyDatabase::jit_stats)
      .def("jit_preload", &PyDatabase::jit_preload, py::arg("max_kernels") = 64);
}
