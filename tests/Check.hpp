#pragma once
// Deliberately tiny: no test-framework dependency to fetch and build.
#include <cmath>
#include <cstdio>
#include <string>

namespace check {
inline int failures = 0;
inline int checks   = 0;

inline void ok(bool cond, const std::string& what) {
  ++checks;
  if (!cond) { ++failures; std::printf("  FAIL  %s\n", what.c_str()); }
  else       { std::printf("  ok    %s\n", what.c_str()); }
}
template <class T>
inline void near(T a, T b, T tol, const std::string& what) {
  ++checks;
  const double e = std::abs(double(a) - double(b));
  if (!(e <= double(tol))) {
    ++failures;
    std::printf("  FAIL  %s   got %.17g want %.17g (err %.3e > tol %.3e)\n",
                what.c_str(), double(a), double(b), e, double(tol));
  } else {
    std::printf("  ok    %s   (err %.3e)\n", what.c_str(), e);
  }
}
inline int report(const char* name) {
  std::printf("[%s] %d checks, %d failures\n", name, checks, failures);
  return failures == 0 ? 0 : 1;
}
}  // namespace check
