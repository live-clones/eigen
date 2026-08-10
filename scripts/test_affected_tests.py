#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for scripts/affected_tests.py.

Runs against a synthetic source tree so the expectations do not drift as the
real headers change, plus a few assertions against the checked-out tree that
only depend on properties the selector must always hold.

Usage: python3 scripts/test_affected_tests.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from affected_tests import IncludeGraph, Selection, select, target_name, test_sources

FIXTURE = {
    "Eigen/Core": '#include "src/Core/util/Meta.h"\n#include "src/Core/Block.h"\n',
    "Eigen/Dense": '#include "Core"\n#include "SVD"\n',
    "Eigen/SVD": '#include "Core"\n#include "src/SVD/BDCSVD.h"\n',
    "Eigen/src/Core/util/Meta.h": "",
    "Eigen/src/Core/Block.h": "",
    "Eigen/src/SVD/BDCSVD.h": "",
    "Eigen/src/Geometry/Quaternion.h": "",
    "test/main.h": "#include <Eigen/Core>\n",
    "test/block.cpp": '#include "main.h"\n',
    "test/bdcsvd.cpp": '#include "main.h"\n#include <Eigen/SVD>\n',
    "test/dense.cpp": '#include "main.h"\n#include <Eigen/Dense>\n',
    "unsupported/test/extra.cpp": '#include "../../test/main.h"\n',
}


def build_fixture(root):
    for rel, content in FIXTURE.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            handle.write(content)


FAILURES = []


def check(condition, message):
    if condition:
        return
    FAILURES.append(message)
    print("FAIL: %s" % message)


def targets_of(selection):
    return sorted(selection.targets)


def test_fixture_graph(root):
    graph = IncludeGraph(root)
    sources = test_sources(graph)
    check(sources == ["test/bdcsvd.cpp", "test/block.cpp", "test/dense.cpp",
                      "unsupported/test/extra.cpp"],
          "test sources discovered in both trees, got %s" % sources)

    # A leaf header reaches only the tests whose closure includes it.
    sel = select(graph, ["Eigen/src/SVD/BDCSVD.h"])
    check(sel.mode == "targets" and targets_of(sel) == ["bdcsvd", "dense"],
          "BDCSVD.h selects bdcsvd and dense, got %s (%s)" % (targets_of(sel), sel.mode))

    # A hub header reaches everything and degrades to the full suite.
    sel = select(graph, ["Eigen/src/Core/util/Meta.h"])
    check(sel.mode == "all", "Meta.h degrades to the full suite, got %s" % sel.mode)

    # ... but not when the threshold allows the explicit list.
    sel = select(graph, ["Eigen/src/Core/util/Meta.h"], max_fraction=1.0)
    check(sel.mode == "targets" and len(sel.targets) == 4,
          "Meta.h reaches all four tests, got %s" % targets_of(sel))

    # Umbrella indirection is followed: Dense -> SVD -> BDCSVD.h.
    sel = select(graph, ["Eigen/Dense"])
    check(sel.mode == "targets" and targets_of(sel) == ["dense"],
          "Eigen/Dense selects only the test including it, got %s" % targets_of(sel))

    # A header that no test reaches selects nothing.
    sel = select(graph, ["Eigen/src/Geometry/Quaternion.h"])
    check(sel.mode == "none", "an unreached header selects nothing, got %s" % sel.mode)

    # A changed test source selects itself.
    sel = select(graph, ["test/block.cpp"])
    check(sel.mode == "targets" and targets_of(sel) == ["block"],
          "a changed test selects itself, got %s" % targets_of(sel))

    # Documentation, benchmarks and metadata select nothing, including their
    # own CMakeLists.txt -- which must not trip the full-rebuild rule.
    sel = select(graph, ["doc/TopicLazyEvaluation.dox", "README.md", ".agents/ci.md",
                         "benchmarks/Core/bench_reductions.cpp",
                         "unsupported/benchmarks/GPU/CMakeLists.txt",
                         "doc/CMakeLists.txt", "failtest/bdcsvd_int.cpp",
                         "debug/gdb/printers.py"])
    check(sel.mode == "none", "docs and benchmarks select nothing, got %s (%s)"
          % (sel.mode, sel.reasons))

    # CMake and CI changes invalidate the mapping.
    for path in ["CMakeLists.txt", "test/CMakeLists.txt", "cmake/EigenTesting.cmake",
                 "ci/scripts/build.linux.script.sh", ".gitlab-ci.yml", "blas/level3_impl.h"]:
        sel = select(graph, [path])
        check(sel.mode == "all", "%s forces the full suite, got %s" % (path, sel.mode))

    # An unknown path (deleted or renamed away) is not guessed at.
    sel = select(graph, ["Eigen/src/Core/util/Removed.h"])
    check(sel.mode == "all", "an unknown path forces the full suite, got %s" % sel.mode)

    # A new test source that exists but is not yet included anywhere.
    new_test = os.path.join(root, "test", "brand_new.cpp")
    with open(new_test, "w") as handle:
        handle.write('#include "main.h"\n')
    graph = IncludeGraph(root)
    sel = select(graph, ["test/brand_new.cpp"])
    check(sel.mode == "targets" and "brand_new" in sel.targets,
          "a new test source is selected, got %s" % targets_of(sel))
    os.remove(new_test)

    # Mixed changes union their selections.
    graph = IncludeGraph(root)
    sel = select(graph, ["Eigen/src/SVD/BDCSVD.h", "unsupported/test/extra.cpp"])
    check(sel.mode == "targets" and targets_of(sel) == ["bdcsvd", "dense", "extra"],
          "mixed changes union, got %s" % targets_of(sel))


def test_output_encoding():
    sel = Selection("targets", ["adjoint", "bdcsvd"])
    check(sel.targets_file == "adjoint\nbdcsvd\n",
          "target list is newline separated, got %r" % sel.targets_file)
    check(sel.regex_file == "^(adjoint|bdcsvd)(_[0-9]+)?$\n",
          "regex matches every part, got %r" % sel.regex_file)

    check(Selection("all").targets_file == "buildtests\n", "full mode builds everything")
    check(Selection("all").regex_file == "ALL\n", "full mode runs everything")
    check(Selection("none").targets_file == "NONE\n", "empty mode builds nothing")
    check(Selection("none").regex_file == "NONE\n", "empty mode runs nothing")

    check(target_name("unsupported/test/GPU/cublas.cpp") == "cublas",
          "target name is the source basename")


def test_real_tree():
    """Properties that must hold against the checked-out tree."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(root, "Eigen", "src")):
        print("skipping real-tree checks: not an Eigen source tree")
        return
    graph = IncludeGraph(root)
    sources = test_sources(graph)
    check(len(sources) > 200, "real tree has many test sources, got %d" % len(sources))

    # main.h is a hub: changing it must run everything.
    sel = select(graph, ["test/main.h"])
    check(sel.mode == "all", "test/main.h runs the full suite, got %s" % sel.mode)

    # A leaf decomposition header must not degrade to the full suite.
    sel = select(graph, ["Eigen/src/Eigenvalues/RealQZ.h"])
    check(sel.mode == "targets", "RealQZ.h yields a selection, got %s" % sel.mode)
    check("real_qz" in sel.targets, "RealQZ.h selects real_qz, got %d targets" % len(sel.targets))

    # The selection must be a superset of what a narrower reading would give.
    sel_svd = select(graph, ["Eigen/src/SVD/BDCSVD.h"])
    check(sel_svd.mode == "targets" and "bdcsvd" in sel_svd.targets,
          "BDCSVD.h selects bdcsvd")

    # Every selected name must be the basename of a real test source.
    basenames = {target_name(s) for s in sources}
    for sel in (sel, sel_svd):
        unknown = sorted(t for t in sel.targets if t not in basenames)
        check(not unknown, "selected names are test sources, got %s" % unknown)


def main():
    root = tempfile.mkdtemp(prefix="eigen-affected-")
    try:
        build_fixture(root)
        test_fixture_graph(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    test_output_encoding()
    test_real_tree()

    if FAILURES:
        print("\n%d check(s) failed" % len(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
