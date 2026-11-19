#include "gpu/jit/jit.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_map>

#include "common/util.h"

#ifndef WARPDB_SRC_DIR
#define WARPDB_SRC_DIR "src"
#endif

namespace warpdb::gpu {

class JitKernel {
 public:
  CUmodule module = nullptr;
  CUfunction fn = nullptr;
};

namespace {

uint64_t fnv1a(const std::string& s, uint64_t h = 1469598103934665603ULL) {
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return h;
}

std::string cu_error(CUresult r) {
  const char* msg = nullptr;
  cuGetErrorString(r, &msg);
  return msg ? msg : "unknown CUDA driver error";
}

}  // namespace

struct JitRuntime::Impl {
  std::mutex mu;
  bool probed = false, ok = false;
  std::string arch, nvrtc_version, disk_dir;
  std::unordered_map<std::string, std::shared_ptr<JitKernel>> cache;
  std::unordered_map<std::string, std::string> failed;  // source key -> NVRTC error (never recompile a bad kernel)
  JitStats stats;

  void probe() {
    if (probed) return;
    probed = true;
    try {
      if (cudaFree(nullptr) != cudaSuccess) return;  // creates + binds the primary context for this thread
      CUcontext ctx = nullptr;
      if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || !ctx) return;
      int major = 0, minor = 0, v1 = 0, v2 = 0;
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
      if (nvrtcVersion(&v1, &v2) != NVRTC_SUCCESS) return;
      arch = "sm_" + std::to_string(major) + std::to_string(minor);
      nvrtc_version = std::to_string(v1) + "." + std::to_string(v2);
      ok = true;
      const char* appdata = std::getenv("LOCALAPPDATA");
      if (disk_dir.empty() && appdata) disk_dir = std::string(appdata) + "\\warpdb\\jit";
    } catch (...) {
      ok = false;
    }
  }

  std::vector<char> compile(const std::string& source) {
    nvrtcProgram prog = nullptr;
    if (nvrtcCreateProgram(&prog, source.c_str(), "warpdb_jit.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS)
      throw JitError("nvrtcCreateProgram failed");
    std::string inc = std::string("-I") + WARPDB_SRC_DIR;
    std::string arch_opt = "--gpu-architecture=" + arch;
    const char* opts[] = {arch_opt.c_str(), "-std=c++17", inc.c_str(), "--extra-device-vectorization"};
    nvrtcResult r = nvrtcCompileProgram(prog, 4, opts);
    size_t log_size = 0;
    nvrtcGetProgramLogSize(prog, &log_size);
    std::string log(log_size, '\0');
    if (log_size > 1) nvrtcGetProgramLog(prog, log.data());
    if (r != NVRTC_SUCCESS) {
      nvrtcDestroyProgram(&prog);
      throw JitError("NVRTC compilation failed:\n" + log);
    }
    size_t cubin_size = 0;
    nvrtcGetCUBINSize(prog, &cubin_size);
    std::vector<char> cubin(cubin_size);
    nvrtcGetCUBIN(prog, cubin.data());
    nvrtcDestroyProgram(&prog);
    return cubin;
  }

  std::shared_ptr<JitKernel> load(const std::vector<char>& cubin, const std::string& entry) {
    auto k = std::make_shared<JitKernel>();
    CUresult r = cuModuleLoadData(&k->module, cubin.data());
    if (r != CUDA_SUCCESS) throw JitError("cuModuleLoadData failed: " + cu_error(r));
    r = cuModuleGetFunction(&k->fn, k->module, entry.c_str());
    if (r != CUDA_SUCCESS) throw JitError("kernel '" + entry + "' not found in module: " + cu_error(r));
    return k;
  }
};

JitRuntime::JitRuntime() : impl_(std::make_unique<Impl>()) {}

JitRuntime& JitRuntime::instance() {
  static JitRuntime* r = new JitRuntime();  // immortal: kernels may be released during process teardown
  return *r;
}

bool JitRuntime::available() {
  std::lock_guard<std::mutex> g(impl_->mu);
  impl_->probe();
  return impl_->ok;
}

JitStats JitRuntime::stats() const { return impl_->stats; }

void JitRuntime::set_disk_cache_dir(const std::string& dir) {
  std::lock_guard<std::mutex> g(impl_->mu);
  impl_->probe();
  impl_->disk_dir = dir;
}

void JitRuntime::clear_memory_cache() {
  std::lock_guard<std::mutex> g(impl_->mu);
  impl_->cache.clear();
}

bool JitRuntime::peek(const std::string& source, const std::string& entry) {
  std::lock_guard<std::mutex> g(impl_->mu);
  impl_->probe();
  if (!impl_->ok) return false;
  uint64_t h = fnv1a(source, fnv1a(impl_->arch + "|" + impl_->nvrtc_version + "|" + entry));
  char hex[40];
  std::snprintf(hex, sizeof hex, "%016llx_%zx", static_cast<unsigned long long>(h), source.size());
  if (impl_->cache.count(hex)) return true;
  return !impl_->disk_dir.empty() && std::filesystem::exists(impl_->disk_dir + "\\" + hex + ".cubin");
}

std::shared_ptr<JitKernel> JitRuntime::get(const std::string& source, const std::string& entry, double* ms,
                                           const char** origin) {
  std::lock_guard<std::mutex> g(impl_->mu);
  impl_->probe();
  // The driver API works on the CALLING THREAD's current context. Hosts such as Streamlit run every request on a
  // different thread, so the context created while probing is not necessarily current here: bind it (cheap).
  cudaFree(nullptr);
  if (!impl_->ok) throw JitError("NVRTC / CUDA driver API not available");
  Timer t;
  uint64_t h = fnv1a(source, fnv1a(impl_->arch + "|" + impl_->nvrtc_version + "|" + entry));
  char hex[40];
  std::snprintf(hex, sizeof hex, "%016llx_%zx", static_cast<unsigned long long>(h), source.size());
  std::string key = hex;

  if (auto it = impl_->cache.find(key); it != impl_->cache.end()) {
    impl_->stats.mem_hits++;
    if (ms) *ms = 0;
    if (origin) *origin = "memory cache";
    return it->second;
  }
  if (auto f = impl_->failed.find(key); f != impl_->failed.end()) throw JitError(f->second);
  std::string path;
  if (!impl_->disk_dir.empty()) path = impl_->disk_dir + "\\" + key + ".cubin";
  try {
    std::vector<char> cubin;
    bool from_disk = false;
    if (!path.empty()) {
      std::ifstream f(path, std::ios::binary | std::ios::ate);
      if (f) {
        cubin.resize(static_cast<size_t>(f.tellg()));
        f.seekg(0);
        f.read(cubin.data(), static_cast<std::streamsize>(cubin.size()));
        from_disk = !cubin.empty();
      }
    }
    double compile_ms = 0;
    if (!from_disk) {
      Timer tc;
      cubin = impl_->compile(source);
      compile_ms = tc.ms();
      impl_->stats.compiles++;
      impl_->stats.compile_ms += compile_ms;
      if (!path.empty()) {
        try {
          std::filesystem::create_directories(impl_->disk_dir);
          std::ofstream(path, std::ios::binary).write(cubin.data(), static_cast<std::streamsize>(cubin.size()));
        } catch (...) {  // cache is best-effort
        }
      }
    } else {
      impl_->stats.disk_hits++;
    }
    Timer tl;
    std::shared_ptr<JitKernel> k;
    try {
      k = impl_->load(cubin, entry);
    } catch (const JitError&) {
      if (from_disk && !path.empty()) std::filesystem::remove(path);  // corrupt / stale CUBIN: rebuild next time
      throw;
    }
    impl_->stats.load_ms += tl.ms();
    impl_->cache[key] = k;
    if (ms) *ms = t.ms();
    if (origin) *origin = from_disk ? "disk cache" : "compiled";
    return k;
  } catch (const JitError& e) {
    impl_->stats.failures++;
    if (std::string(e.what()).rfind("NVRTC compilation failed", 0) == 0) impl_->failed[key] = e.what();
    throw;
  }
}

void JitRuntime::launch(JitKernel& k, unsigned grid, unsigned block, ArgPack& args) {
  if (grid == 0) return;
  cudaFree(nullptr);  // see get(): make the primary context current on this thread
  CUresult r = cuLaunchKernel(k.fn, grid, 1, 1, block, 1, 1, 0, nullptr, args.data(), nullptr);
  if (r != CUDA_SUCCESS) throw JitError("cuLaunchKernel failed: " + cu_error(r));
}

}  // namespace warpdb::gpu
