#pragma once

// Server log lines: one event per line, the request id first ("req 42 ...")
// so `grep "req 42 "` gives one request's story. Under systemd (stderr connected to the journal: $JOURNAL_STREAM names
// stderr's device:inode - a terminal started from a desktop session inherits the variable, so it alone isn't enough)
// each line starts with a syslog priority prefix "<N>", which journald strips and stores as the entry's priority -
// journalctl then colours errors red and warnings yellow, and `journalctl -p info` hides the debug detail lines.
// Run by hand, lines are plain text.

#include <sys/stat.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

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

}  // namespace strix
