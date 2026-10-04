#!/usr/bin/env python3
"""Per-kernel summary of a rocprofv3 trace (the *_results.db file from `--output-format rocpd`).

Prints one line per kernel: how many times it ran, its average and total GPU time, and its launch shape - how many
workgroups it was split into. Give it two traces to compare them side by side (before / after a change).

Usage: trace_summary.py TRACE.db [OTHER_TRACE.db]
"""
import os
import sqlite3
import sys


def summarize(path):
    if not os.path.isfile(path):
        sys.exit(f"trace_summary: {path} not found (expected the *_results.db that rocprofv3 wrote)")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type in ('table', 'view')")}
    if "kernels" not in tables:
        sys.exit(f"trace_summary: {path} has no 'kernels' view - was it captured with --kernel-trace "
                 f"--output-format rocpd?")
    rows = con.execute("""
        select name, count(*), avg("end" - start) / 1000.0, sum("end" - start) / 1000.0,
               max(grid_x / workgroup_x) * max(grid_y / workgroup_y) * max(grid_z / workgroup_z)
        from kernels group by name order by sum("end" - start) desc""").fetchall()
    con.close()
    return {r[0]: r[1:] for r in rows}


def short(name):
    return name.split("(")[0]


def main(argv):
    if len(argv) not in (2, 3):
        sys.exit(__doc__)
    a = summarize(argv[1])
    if len(argv) == 2:
        print(f"{'kernel':34} {'calls':>5} {'avg us':>9} {'total us':>10} {'workgroups':>10}")
        for name, (calls, avg, total, groups) in a.items():
            print(f"{short(name)[:34]:34} {calls:>5} {avg:>9.2f} {total:>10.1f} {groups:>10}")
        return
    b = summarize(argv[2])
    print(f"{'kernel':34} {'avg us (1)':>11} {'avg us (2)':>11} {'change':>8}")
    for name in sorted(set(a) | set(b), key=lambda n: -max(a.get(n, (0, 0, 0))[2], b.get(n, (0, 0, 0))[2])):
        x, y = a.get(name), b.get(name)
        ax = f"{x[1]:.2f}" if x else "-"
        by = f"{y[1]:.2f}" if y else "-"
        change = f"{(y[1] / x[1] - 1) * 100:+.0f}%" if x and y else ""
        print(f"{short(name)[:34]:34} {ax:>11} {by:>11} {change:>8}")


if __name__ == "__main__":
    main(sys.argv)
