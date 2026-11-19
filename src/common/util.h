#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace warpdb {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline unsigned num_threads() {
  static unsigned n = std::max(1u, std::thread::hardware_concurrency());
  return n;
}

// Number of chunks parallel_for will use for (n, grain). Use it to size per-chunk state.
inline size_t num_chunks(size_t n, size_t grain) {
  if (n == 0) return 0;
  size_t t = num_threads();
  if (n <= grain || t == 1) return 1;
  return std::min((n + grain - 1) / grain, t * 4);
}

// f(chunk, begin, end). Chunks are contiguous and ordered by index, so concatenating
// per-chunk results in chunk order preserves row order. Exceptions are propagated.
template <class F>
void parallel_for(size_t n, size_t grain, F&& f) {
  size_t chunks = num_chunks(n, grain);
  if (chunks == 0) return;
  if (chunks == 1) {
    f(size_t(0), size_t(0), n);
    return;
  }
  size_t csize = (n + chunks - 1) / chunks;
  std::atomic<size_t> next{0};
  std::exception_ptr err;
  std::mutex mu;
  auto worker = [&] {
    try {
      for (;;) {
        size_t c = next.fetch_add(1);
        if (c >= chunks) break;
        size_t b = c * csize, e = std::min(n, b + csize);
        if (b < e) f(c, b, e);
      }
    } catch (...) {
      std::lock_guard<std::mutex> g(mu);
      if (!err) err = std::current_exception();
    }
  };
  unsigned nt = static_cast<unsigned>(std::min<size_t>(num_threads(), chunks));
  std::vector<std::thread> th;
  for (unsigned i = 1; i < nt; ++i) th.emplace_back(worker);
  worker();
  for (auto& t : th) t.join();
  if (err) std::rethrow_exception(err);
}

struct Timer {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  void reset() { t0 = std::chrono::steady_clock::now(); }
};

// ---- dates (days since 1970-01-01), Howard Hinnant's civil algorithms ----
inline int32_t days_from_civil(int y, unsigned m, unsigned d) {
  y -= m <= 2;
  const int era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<int>(doe) - 719468;
}

inline void civil_from_days(int32_t z, int& y, unsigned& m, unsigned& d) {
  z += 719468;
  const int era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  y = static_cast<int>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  d = doy - (153 * mp + 2) / 5 + 1;
  m = mp < 10 ? mp + 3 : mp - 9;
  y += (m <= 2);
}

inline unsigned days_in_month(int y, unsigned m) {
  static const unsigned dm[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if (m == 2 && ((y % 4 == 0 && y % 100 != 0) || y % 400 == 0)) return 29;
  return dm[m - 1];
}

// Strict yyyy-mm-dd.
inline bool parse_date(std::string_view s, int32_t& out) {
  if (s.size() != 10 || s[4] != '-' || s[7] != '-') return false;
  auto dig = [&](int i) { return s[i] >= '0' && s[i] <= '9'; };
  for (int i : {0, 1, 2, 3, 5, 6, 8, 9})
    if (!dig(i)) return false;
  int y = (s[0] - '0') * 1000 + (s[1] - '0') * 100 + (s[2] - '0') * 10 + (s[3] - '0');
  unsigned m = (s[5] - '0') * 10 + (s[6] - '0');
  unsigned d = (s[8] - '0') * 10 + (s[9] - '0');
  if (m < 1 || m > 12 || d < 1 || d > days_in_month(y, m)) return false;
  out = days_from_civil(y, m, d);
  return true;
}

inline std::string format_date(int32_t days) {
  int y;
  unsigned m, d;
  civil_from_days(days, y, m, d);
  char buf[16];
  std::snprintf(buf, sizeof buf, "%04d-%02u-%02u", y, m, d);
  return buf;
}

inline std::string to_lower(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

}  // namespace warpdb
