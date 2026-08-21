#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""A stand-in for a compiled comparison benchmark executable.

It implements exactly the Google Benchmark command-line surface `run.py` drives
-- `--benchmark_list_tests`, `--benchmark_filter`, `--benchmark_out`,
`--benchmark_out_format`, `--benchmark_repetitions`, `--benchmark_min_time`,
`--benchmark_report_aggregates_only` -- and replays a canned JSON document, so
the harness tests never need a built binary, a BLAS, or a quiet machine.

Two files sit beside it:

* `canned_benchmark_output.json` (or `$STUB_BENCH_JSON`): the document to
  replay.  Rows are filtered by `--benchmark_filter` against `run_name`.
* `$STUB_BENCH_TRACE`, when set: one JSON object per invocation recording argv
  and the whole environment, which is how the tests assert that every vendor
  thread-count variable really reached the child process.

`$STUB_BENCH_FAIL_RC`, when set, makes the stub exit with that code after
writing the trace and nothing else, standing in for a benchmark that died.

`$STUB_BENCH_ERROR_RC` is the other failure the real binaries have: everything is
written normally and THEN the process exits non-zero.  That is what
`ErrorTrackingReporter` in bench_compare.h does after an unstructured
`SkipWithError` -- Google Benchmark has already emitted every row, and the status
reports that one of them disagreed with Eigen.
"""

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def canned():
    path = os.environ.get("STUB_BENCH_JSON") or str(HERE / "canned_benchmark_output.json")
    return json.loads(Path(path).read_text())


def trace(argv):
    path = os.environ.get("STUB_BENCH_TRACE")
    if not path:
        return
    record = {"argv": list(argv), "env": dict(os.environ), "cwd": os.getcwd()}
    with open(path, "a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def option(argv, name):
    prefix = name + "="
    for arg in argv:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == name:
            index = argv.index(arg)
            if index + 1 < len(argv):
                return argv[index + 1]
    return None


def main(argv):
    trace(argv)

    failure = os.environ.get("STUB_BENCH_FAIL_RC")
    if failure:
        sys.stderr.write("stub_benchmark: forced failure\n")
        return int(failure)

    document = canned()
    pattern = option(argv, "--benchmark_filter")
    rows = document.get("benchmarks", [])
    if pattern:
        matcher = re.compile(pattern)
        rows = [row for row in rows if matcher.search(row.get("run_name", row.get("name", "")))]
    document["benchmarks"] = rows

    if "--benchmark_list_tests" in argv or option(argv, "--benchmark_list_tests") in ("true", "1"):
        names = []
        for row in rows:
            name = row.get("run_name", row.get("name", ""))
            if name and name not in names:
                names.append(name)
        sys.stdout.write("".join(name + "\n" for name in names))
        return 0

    out = option(argv, "--benchmark_out")
    text = json.dumps(document, indent=2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)
        sys.stderr.write(f"stub_benchmark: wrote {len(rows)} rows to {out}\n")
    else:
        sys.stdout.write(text)

    errored = os.environ.get("STUB_BENCH_ERROR_RC")
    if errored:
        sys.stderr.write("stub_benchmark: a benchmark reported an error\n")
        return int(errored)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
