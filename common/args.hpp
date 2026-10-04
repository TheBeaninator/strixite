#pragma once

// Checked command-line parsing for tools/tests: a bad argument is an error
// naming the argument and the accepted range, never a silent atoi() zero.

#include "common/check.hpp"

#include <cerrno>
#include <cstdlib>
#include <string>

namespace strix {

inline long long parse_int(const std::string &s, const std::string &what, long long lo, long long hi) {
    STRIX_CHECK(!s.empty(), what, " is empty; expected an integer in [", lo, ", ", hi, "]");
    char *end = nullptr;
    errno = 0;
    long long v = std::strtoll(s.c_str(), &end, 10);
    STRIX_CHECK(errno == 0 && end == s.c_str() + s.size(), what, " = \"", s, "\" is not an integer");
    STRIX_CHECK(v >= lo && v <= hi, what, " = ", v, " is out of range [", lo, ", ", hi, "]");
    return v;
}

inline double parse_double(const std::string &s, const std::string &what, double lo, double hi) {
    STRIX_CHECK(!s.empty(), what, " is empty; expected a number in [", lo, ", ", hi, "]");
    char *end = nullptr;
    errno = 0;
    double v = std::strtod(s.c_str(), &end);
    STRIX_CHECK(errno == 0 && end == s.c_str() + s.size(), what, " = \"", s, "\" is not a number");
    STRIX_CHECK(v >= lo && v <= hi, what, " = ", v, " is out of range [", lo, ", ", hi, "]");
    return v;
}

}  // namespace strix
