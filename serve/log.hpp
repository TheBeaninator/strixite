#pragma once

// Server log lines. Each request is one block, printed by
// the engine thread, which runs one request at a time - so a block's lines are never mixed with another request's:
//
//   ----------------------------------------
//   Request 347
//   from        127.0.0.1, POST /v1/chat/completions, 95.7 KB, not streamed
//   ...
//   prefill         1,485 t/s   17,655 new tokens in 11.89 s
//   generate          52.8 t/s   269 tokens (think 0) in 5.07 s
//   mtp         1,148/1,374 drafts accepted (83.6%), 1.9 tokens per forward
//               121 rollbacks; accepted before the reject 0/1/2/3+: 59/24/13/25
//   done             44.06 s total, 0.54 s first token, stop, queue 0.00 s
//   ----------------------------------------
//
// The rows read for speed (prefill, generate) lead with their rate, right-aligned in one column, so it scans down;
// counts and rates carry comma thousands separators (fmt_n, fmt_rate - always commas, never a locale's).
// A row is a label in a fixed column and its text (slog_row); a long row goes on under an empty label. A warning row's
// text starts with "WARNING:". Lines from other threads stay single events with their own prefix: the prompt cache's
// writer ("prompt cache: ..."), a request that waits ("Request 348 queued ..."), one refused before the engine
// ("Request 350 rejected: ..."). One request's block: tools/reqlog.sh 347.
//
// Under systemd (stderr connected to the journal: $JOURNAL_STREAM names
// stderr's device:inode - a terminal started from a desktop session inherits the variable, so it alone isn't enough)
// each line starts with a syslog priority prefix "<N>", which journald strips and stores as the entry's priority -
// journalctl then colours errors red and warnings yellow, and `journalctl -p info` hides the debug detail lines.
// Run by hand, lines are plain text.

#include <sys/stat.h>

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "common/check.hpp"

namespace strix {

enum class LogLevel { Error = 3, Warning = 4, Info = 6, Debug = 7 };

inline bool log_to_journal() {
    static const bool on = [] {
        const char *js = std::getenv("JOURNAL_STREAM");
        unsigned long long dev = 0, ino = 0;
        struct stat st {};
        return js && std::sscanf(js, "%llu:%llu", &dev, &ino) == 2 && ::fstat(2, &st) == 0 &&
               (unsigned long long)st.st_dev == dev && (unsigned long long)st.st_ino == ino;
    }();
    return on;
}

// printf-style; one line (a trailing newline is added).
#if defined(__GNUC__)
__attribute__((format(printf, 2, 3)))
#endif
inline void slog(LogLevel level, const char *fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (log_to_journal()) std::fprintf(stderr, "<%d>%s\n", (int)level, buf);
    else std::fprintf(stderr, "%s\n", buf);
}

// A request block's first and last line.
constexpr const char *kLogRule = "----------------------------------------";
// A row's label column: labels are at most kLogLabelWidth - 2 characters, so at least two spaces follow.
constexpr int kLogLabelWidth = 12;

// One row of a request block: `label` (empty: the row above goes on) in the label column, then the printf-style
// text.
#if defined(__GNUC__)
__attribute__((format(printf, 3, 4)))
#endif
inline void slog_row(LogLevel level, const char *label, const char *fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    slog(level, "%-*s%s", kLogLabelWidth, label ? label : "", buf);
}

// Number formats for log text: counts and rates with comma thousands separators - "286,816",
// "1,485.4" - so long numbers read at a glance. Always commas, never the locale's separator (setlocale would hijack
// printf's "%'ld" and is absent on some systems anyway): the journal stays grep-able on any machine.

namespace detail {

// Inserts a comma every 3 digits: "1234567" -> "1,234,567". `digits` is plain digits; `negative` re-prepends the sign.
inline std::string group_thousands(const std::string &digits, bool negative) {
    STRIX_CHECK(!digits.empty() && digits.find_first_not_of("0123456789") == std::string::npos,
                "group_thousands: not plain digits: '", digits, "'");
    std::string out = negative ? "-" : "";
    for (size_t i = 0; i < digits.size();) {
        const size_t chunk = i == 0 ? (digits.size() % 3 ? digits.size() % 3 : 3) : 3;
        if (i) out += ',';
        out.append(digits, i, chunk);
        i += chunk;
    }
    return out;
}

}  // namespace detail

// A count: 286816 -> "286,816", -1234567 -> "-1,234,567".
inline std::string fmt_n(long long count) {
    char buf[32];
    const int n = std::snprintf(buf, sizeof buf, "%lld", count);
    STRIX_CHECK(n > 0 && n < (int)sizeof buf, "fmt_n: formatting ", count, " failed");
    const bool neg = buf[0] == '-';
    return detail::group_thousands(neg ? buf + 1 : buf, neg);
}

// A rate: `decimals` fixed decimals, comma-grouped whole part: (1485.4, 1) -> "1,485.4".
inline std::string fmt_rate(double value, int decimals) {
    STRIX_CHECK(decimals >= 0 && decimals <= 9, "fmt_rate: decimals ", decimals, ", expected 0-9");
    STRIX_CHECK(std::isfinite(value), "fmt_rate: value ", value, " isn't finite");
    char buf[512];
    const int n = std::snprintf(buf, sizeof buf, "%.*f", decimals, value);
    STRIX_CHECK(n > 0 && n < (int)sizeof buf, "fmt_rate: formatting ", value, " failed");
    const std::string s = buf;
    const bool neg = s[0] == '-';
    const size_t dot = s.find('.');
    const size_t whole_begin = neg ? 1 : 0;
    const std::string whole = dot == std::string::npos ? s.substr(whole_begin) : s.substr(whole_begin, dot - whole_begin);
    const std::string frac = dot == std::string::npos ? "" : s.substr(dot);
    return detail::group_thousands(whole, neg) + frac;
}

}  // namespace strix
