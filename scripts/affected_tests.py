#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Select the tests affected by a set of changed files.

Eigen is header-only, so a test is affected by a change exactly when its
translation unit textually includes the changed file.  This script builds the
include graph over ``Eigen/``, ``unsupported/Eigen/`` and the test trees, then
maps changed paths to the CMake test targets that reach them.

The include closure is a strict superset of the true compile dependency: it
follows every ``#include`` regardless of preprocessor guards, so a test is
never dropped because a conditional branch was not taken.  Over-approximation
is the safe direction here -- the point is to widen coverage relative to the
fixed smoke list, not to minimise work.

Changes that invalidate the mapping itself (CMake, CI, the BLAS/LAPACK shims)
fall back to the full ``buildtests`` target rather than a selection.

Two output files are written, both consumed by ``ci/scripts/build.linux.script.sh``
and ``ci/scripts/test.linux.script.sh``:

  targets.txt       ``buildtests``, ``NONE``, or a newline-separated target list
  ctest_regex.txt   ``ALL``, ``NONE``, or a CTest ``-R`` regex

The selected names are CMake target names, not CTest test names: a split test
``foo`` registers ``foo_1``..``foo_N`` as tests but a single ``foo`` target that
aggregates them, so selecting ``foo`` builds and runs every part.  Targets that
do not exist in a given configuration (optional dependencies such as CHOLMOD or
SYCL) are filtered out by the build script, which is the only place that knows
what CMake actually configured.
"""

import argparse
import fnmatch
import os
import re
import subprocess
import sys

# Directories scanned to build the include graph.
SCAN_ROOTS = ("Eigen", "unsupported/Eigen", "test", "unsupported/test")

# Directories whose .cpp files are test translation units.
TEST_ROOTS = ("test", "unsupported/test")

# Changes matching these patterns cannot affect which tests exist or what they
# cover, so they select nothing.
IGNORED_PATTERNS = (
    ".gitattributes",
    ".gitignore",
    ".clang-format",
    ".clang-tidy",
    "*.md",
    "*.dox",
    "AGENTS.md",
    "COPYING*",
    "INSTALL",
    "README*",
    "REUSE.toml",
    ".agents/*",
    ".gitlab/*",
    "LICENSES/*",
    "benchmarks/*",
    "debug/*",
    "demos/*",
    "doc/*",
    "failtest/*",
    "unsupported/benchmarks/*",
    "unsupported/doc/*",
)

# Changes matching these patterns invalidate the include-graph mapping itself
# (test registration, split counts, the CI drivers, or shim libraries whose
# tests are not modelled here), so they force the full test suite.  Checked
# after IGNORED_PATTERNS, so a benchmark's or the docs' own CMakeLists.txt does
# not drag in the whole suite.
FULL_REBUILD_PATTERNS = (
    "CMakeLists.txt",
    "*/CMakeLists.txt",
    "*.cmake",
    "*.cmake.in",
    ".gitlab-ci.yml",
    "ci/*",
    "cmake/*",
    "scripts/*",
    "blas/*",
    "lapack/*",
)

INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]', re.MULTILINE)


def _matches(path, patterns):
    return any(fnmatch.fnmatch(path, p) for p in patterns)


class IncludeGraph:
    """Textual ``#include`` graph over the scanned source roots."""

    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.files = set()
        self._direct = {}
        self._by_suffix = {}
        self._scan()

    def _scan(self):
        for root in SCAN_ROOTS:
            abs_root = os.path.join(self.source_dir, root)
            if not os.path.isdir(abs_root):
                continue
            for dirpath, dirnames, filenames in os.walk(abs_root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for name in filenames:
                    rel = os.path.relpath(os.path.join(dirpath, name), self.source_dir)
                    self.files.add(rel)
        # Index every path suffix so that an include spelled relative to a
        # directory outside the scanned roots still resolves.  Ambiguous
        # suffixes are dropped rather than guessed.
        counts = {}
        for rel in self.files:
            parts = rel.split("/")
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                counts.setdefault(suffix, []).append(rel)
        self._by_suffix = {k: v[0] for k, v in counts.items() if len(v) == 1}

    def _read(self, rel):
        try:
            with open(os.path.join(self.source_dir, rel), "r", errors="ignore") as handle:
                return handle.read()
        except OSError:
            return ""

    def direct_includes(self, rel):
        """Resolved includes of a single file."""
        cached = self._direct.get(rel)
        if cached is not None:
            return cached
        resolved = set()
        self._direct[rel] = resolved  # placed first: the graph has cycles
        directory = os.path.dirname(rel)
        for spelling in INCLUDE_RE.findall(self._read(rel)):
            candidate = os.path.normpath(os.path.join(directory, spelling))
            if candidate in self.files:
                resolved.add(candidate)
            elif spelling in self.files:
                resolved.add(spelling)
            elif spelling in self._by_suffix:
                resolved.add(self._by_suffix[spelling])
        return resolved

    def closure(self, rel):
        """Every file reachable from ``rel`` through includes."""
        seen = set()
        stack = [rel]
        while stack:
            for nxt in self.direct_includes(stack.pop()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen


def test_sources(graph):
    """Test translation units, as repo-relative paths."""
    return sorted(
        rel
        for rel in graph.files
        if rel.endswith(".cpp")
        and any(rel.startswith(root + "/") for root in TEST_ROOTS)
    )


def target_name(test_source):
    """CMake target aggregating every part of a test source."""
    return os.path.splitext(os.path.basename(test_source))[0]


def reverse_map(graph, sources):
    """Map each included file to the test sources that reach it."""
    reverse = {}
    for src in sources:
        for dep in graph.closure(src):
            reverse.setdefault(dep, set()).add(src)
    return reverse


class Selection:
    """Outcome of a selection: either the full suite or an explicit target set."""

    def __init__(self, mode, targets=(), reasons=()):
        self.mode = mode  # "all", "targets", or "none"
        self.targets = set(targets)
        self.reasons = list(reasons)

    @property
    def targets_file(self):
        if self.mode == "all":
            return "buildtests\n"
        if self.mode == "none":
            return "NONE\n"
        return "".join(name + "\n" for name in sorted(self.targets))

    @property
    def regex_file(self):
        if self.mode == "all":
            return "ALL\n"
        if self.mode == "none":
            return "NONE\n"
        alternatives = "|".join(re.escape(name) for name in sorted(self.targets))
        return "^(%s)(_[0-9]+)?$\n" % alternatives


def select(graph, changed_files, max_fraction=0.85):
    """Map changed paths to the tests that must run."""
    sources = test_sources(graph)
    reverse = reverse_map(graph, sources)

    selected = set()
    reasons = []
    for path in changed_files:
        path = path.strip()
        if not path:
            continue
        if _matches(path, IGNORED_PATTERNS):
            continue
        if _matches(path, FULL_REBUILD_PATTERNS):
            return Selection("all", reasons=["%s forces the full suite" % path])
        hits = reverse.get(path)
        if hits is not None:
            selected |= hits
            continue
        if path in graph.files:
            # A source file in the tree that nothing includes: a new header not
            # yet wired up, or a test source that was added in this change.
            if any(path.startswith(root + "/") for root in TEST_ROOTS) and path.endswith(".cpp"):
                selected.add(path)
                continue
            reasons.append("%s is in the tree but reaches no test" % path)
            continue
        # Deleted, renamed, or outside every scanned root: the graph cannot say
        # what it affected, so do not guess.
        return Selection("all", reasons=["%s is not in the include graph" % path])

    if not selected:
        return Selection("none", reasons=reasons or ["no change reaches a test"])

    if len(selected) > max_fraction * len(sources):
        reasons.append(
            "%d of %d test sources selected (>%.0f%%)"
            % (len(selected), len(sources), 100 * max_fraction)
        )
        return Selection("all", reasons=reasons)

    return Selection("targets", (target_name(s) for s in selected), reasons)


def changed_files_from_git(source_dir, base_sha, head="HEAD"):
    """Paths changed between ``base_sha`` and ``head``."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "%s...%s" % (base_sha, head)],
        cwd=source_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("git diff failed: %s" % result.stderr.strip())
    return [line for line in result.stdout.splitlines() if line.strip()]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_source = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--source-dir", default=default_source,
                        help="Eigen source tree (default: the tree containing this script)")
    parser.add_argument("--base-sha",
                        help="compute changed files from 'git diff BASE...HEAD'")
    parser.add_argument("--head", default="HEAD", help="head revision for --base-sha")
    parser.add_argument("--changed-files",
                        help="read newline-separated changed paths from this file ('-' for stdin)")
    parser.add_argument("--output-dir",
                        help="write targets.txt and ctest_regex.txt here")
    parser.add_argument("--max-fraction", type=float, default=0.85,
                        help="degrade to the full suite above this fraction (default: 0.85)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.changed_files:
        if args.changed_files == "-":
            changed = sys.stdin.read().splitlines()
        else:
            with open(args.changed_files) as handle:
                changed = handle.read().splitlines()
    elif args.base_sha:
        try:
            changed = changed_files_from_git(args.source_dir, args.base_sha, args.head)
        except RuntimeError as error:
            # Without a usable diff there is no basis for narrowing.
            print("%s; selecting the full suite" % error, file=sys.stderr)
            changed = None
    else:
        print("one of --base-sha or --changed-files is required", file=sys.stderr)
        return 2

    if changed is None:
        selection = Selection("all", reasons=["the merge-base diff is unavailable"])
    else:
        graph = IncludeGraph(args.source_dir)
        selection = select(graph, changed, args.max_fraction)

    print("mode: %s" % selection.mode, file=sys.stderr)
    for reason in selection.reasons:
        print("  %s" % reason, file=sys.stderr)
    if selection.mode == "targets":
        print("  %d targets: %s" % (len(selection.targets),
                                    " ".join(sorted(selection.targets))), file=sys.stderr)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "targets.txt"), "w") as handle:
            handle.write(selection.targets_file)
        with open(os.path.join(args.output_dir, "ctest_regex.txt"), "w") as handle:
            handle.write(selection.regex_file)
    else:
        sys.stdout.write(selection.targets_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
