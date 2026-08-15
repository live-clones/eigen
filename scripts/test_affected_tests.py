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
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from affected_tests import (
    IncludeGraph,
    Selection,
    changed_files_from_git,
    full_suite,
    select,
    test_registrations,
)

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
    "test/multitu.cpp": '#include "main.h"\n',
    "test/multitu_main.cpp": '#include "main.h"\n',
    "test/CMakeLists.txt": """ei_add_test(block)
ei_add_test(bdcsvd)
ei_add_test(dense)
add_executable(multitu multitu.cpp multitu_main.cpp)
""",
    "unsupported/test/extra.cpp": '#include "../../test/main.h"\n',
    "unsupported/test/CMakeLists.txt": "ei_add_test(extra)\n",
    "failtest/svd_int.cpp": "#include <Eigen/SVD>\n",
    "failtest/const_block.cpp": "#include <Eigen/Core>\n",
    "failtest/CMakeLists.txt": 'ei_add_failtest("svd_int")\nei_add_failtest("const_block")\n',
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
    sources = sorted(test_registrations(graph).targets)
    check(sources == ["test/bdcsvd.cpp", "test/block.cpp", "test/dense.cpp",
                      "test/multitu.cpp", "test/multitu_main.cpp",
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
    check(sel.mode == "targets" and len(sel.targets) == 5,
          "Meta.h reaches all five targets, got %s" % targets_of(sel))

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

    # Multi-translation-unit executables map every source to the registered
    # target rather than assuming each basename is a target.
    sel = select(graph, ["test/multitu_main.cpp"])
    check(sel.mode == "targets" and targets_of(sel) == ["multitu"],
          "a secondary translation unit selects its executable, got %s" % targets_of(sel))

    # Documentation, benchmarks and metadata select nothing, including their
    # own CMakeLists.txt -- which must not trip the full-rebuild rule.
    sel = select(graph, ["doc/TopicLazyEvaluation.dox", "README.md", ".agents/ci.md",
                         "benchmarks/Core/bench_reductions.cpp",
                         "unsupported/benchmarks/GPU/CMakeLists.txt",
                         "doc/CMakeLists.txt", "debug/gdb/printers.py"])
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

    # A new test source must not disappear as an unconfigured target if its
    # CMake registration was forgotten.
    new_test = os.path.join(root, "test", "brand_new.cpp")
    with open(new_test, "w") as handle:
        handle.write('#include "main.h"\n')
    graph = IncludeGraph(root)
    sel = select(graph, ["test/brand_new.cpp"])
    check(sel.mode == "error", "an unregistered test source fails selection, got %s" % sel.mode)

    # A full-suite change takes precedence regardless of path order; this is
    # the normal path when a new source and its CMake registration land together.
    sel = select(graph, ["test/brand_new.cpp", "test/CMakeLists.txt"])
    check(sel.mode == "all", "CMake changes force the full suite before source validation")

    with open(os.path.join(root, "test", "CMakeLists.txt"), "a") as handle:
        handle.write("ei_add_test(brand_new)\n")
    graph = IncludeGraph(root)
    sel = select(graph, ["test/brand_new.cpp"])
    check(sel.mode == "targets" and "brand_new" in sel.targets,
          "a registered new test source is selected, got %s" % targets_of(sel))
    os.remove(new_test)

    # Mixed changes union their selections.
    graph = IncludeGraph(root)
    sel = select(graph, ["Eigen/src/SVD/BDCSVD.h", "unsupported/test/extra.cpp"])
    check(sel.mode == "targets" and targets_of(sel) == ["bdcsvd", "dense", "extra"],
          "mixed changes union, got %s" % targets_of(sel))


def test_failtests(root):
    """The compile-failure suite is selected as CTest names, never as targets."""
    graph = IncludeGraph(root)
    registered = test_registrations(graph)
    check(sorted(registered.failtests) == ["failtest/const_block.cpp", "failtest/svd_int.cpp"],
          "failtests are discovered, got %s" % sorted(registered.failtests))
    check(registered.failtests["failtest/svd_int.cpp"] == ("svd_int_ok", "svd_int_ko"),
          "each failtest registers an _ok and a _ko CTest test")
    check(not any(t.startswith("svd_int") for t in registered.targets.values()),
          "failtests are not build targets")

    # A changed failtest source runs itself and nothing else.  It has no build
    # target: CTest compiles it as the test action.
    sel = select(graph, ["failtest/svd_int.cpp"])
    check(sel.mode == "targets" and targets_of(sel) == [],
          "a failtest selects no build target, got %s" % targets_of(sel))
    check(sorted(sel.ctest_names) == ["svd_int_ko", "svd_int_ok"],
          "a failtest selects its CTest names, got %s" % sorted(sel.ctest_names))
    check(sel.targets_file == "", "a failtest-only selection builds nothing, got %r"
          % sel.targets_file)
    check(sel.regex_file == "^(svd_int_ko|svd_int_ok)(_[0-9]+)?$\n",
          "a failtest-only regex names both parts, got %r" % sel.regex_file)

    # A header reaches its failtests through the same include closure as its
    # tests, so both are selected together.
    sel = select(graph, ["Eigen/src/SVD/BDCSVD.h"])
    check(sel.mode == "targets" and targets_of(sel) == ["bdcsvd", "dense"],
          "BDCSVD.h still selects its tests, got %s" % targets_of(sel))
    check(sorted(sel.ctest_names) == ["svd_int_ko", "svd_int_ok"],
          "BDCSVD.h also selects the failtest reaching it, got %s" % sorted(sel.ctest_names))
    check(sel.regex_file == "^(bdcsvd|dense|svd_int_ko|svd_int_ok)(_[0-9]+)?$\n",
          "targets and CTest-only names share one regex, got %r" % sel.regex_file)

    # A failtest with no ei_add_failtest call is an error, like a test source
    # with no registration.
    orphan = os.path.join(root, "failtest", "orphan.cpp")
    with open(orphan, "w") as handle:
        handle.write("#include <Eigen/Core>\n")
    try:
        sel = select(IncludeGraph(root), ["failtest/orphan.cpp"])
        check(sel.mode == "error",
              "an unregistered failtest fails selection, got %s" % sel.mode)
    finally:
        os.remove(orphan)


def test_output_encoding():
    sel = Selection("targets", ["adjoint", "bdcsvd"])
    check(sel.targets_file == "adjoint\nbdcsvd\n",
          "target list is newline separated, got %r" % sel.targets_file)
    check(sel.regex_file == "^(adjoint|bdcsvd)(_[0-9]+)?$\n",
          "regex matches every part, got %r" % sel.regex_file)

    sel = Selection("targets", ["adjoint"], ctest_names=["x_ok", "x_ko"])
    check(sel.targets_file == "adjoint\n",
          "CTest-only names stay out of the target list, got %r" % sel.targets_file)
    check(sel.regex_file == "^(adjoint|x_ko|x_ok)(_[0-9]+)?$\n",
          "CTest-only names join the regex, got %r" % sel.regex_file)

    check(Selection("all").targets_file == "buildtests\n", "full mode builds everything")
    # buildtests does not aggregate a bare add_executable, so full mode has to
    # name those targets or they stop being compiled.
    check(Selection("all", standalone=["bug1213"]).targets_file == "buildtests\nbug1213\n",
          "full mode names the targets buildtests omits, got %r"
          % Selection("all", standalone=["bug1213"]).targets_file)
    check(Selection("all").regex_file == "ALL\n", "full mode runs everything")
    check(Selection("none").targets_file == "NONE\n", "empty mode builds nothing")
    check(Selection("none").regex_file == "NONE\n", "empty mode runs nothing")


def test_git_rename_paths():
    root = tempfile.mkdtemp(prefix="eigen-affected-git-")
    try:
        old_path = os.path.join(root, "Eigen", "src", "Old.h")
        new_path = os.path.join(root, "Eigen", "src", "New.h")
        os.makedirs(os.path.dirname(old_path))
        with open(old_path, "w") as handle:
            handle.write("// test\n")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "Eigen/src/Old.h"], cwd=root, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Eigen Tests", "-c", "user.email=eigen@example.com",
             "commit", "-qm", "base"],
            cwd=root,
            check=True,
        )
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        os.rename(old_path, new_path)
        subprocess.run(
            ["git", "add", "Eigen/src/Old.h", "Eigen/src/New.h"], cwd=root, check=True
        )
        subprocess.run(
            ["git", "-c", "user.name=Eigen Tests", "-c", "user.email=eigen@example.com",
             "commit", "-qm", "rename"],
            cwd=root,
            check=True,
        )
        changed = changed_files_from_git(root, base)
        check(changed == ["Eigen/src/New.h", "Eigen/src/Old.h"],
              "renames expose both paths, got %s" % changed)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_real_tree():
    """Properties that must hold against the checked-out tree."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(root, "Eigen", "src")):
        print("skipping real-tree checks: not an Eigen source tree")
        return
    graph = IncludeGraph(root)
    registered = test_registrations(graph)
    source_targets = registered.targets
    check(len(source_targets) > 200,
          "real tree has many test sources, got %d" % len(source_targets))

    # bug1213 is a manual multi-source add_executable: nothing aggregates it,
    # so the full-suite selection has to name it or the link regression stops
    # being compiled.
    if "test/bug1213.cpp" in graph.files:
        check("bug1213" in registered.standalone,
              "bug1213 is not covered by buildtests, got %s" % sorted(registered.standalone))
        check("\nbug1213\n" in full_suite(graph, []).targets_file,
              "full mode names bug1213, got %r" % full_suite(graph, []).targets_file)
    check(all(t not in registered.standalone for t in ("bdcsvd", "block")),
          "ei_add_test targets are aggregated by buildtests")

    # The compile-failure suite must stay reachable: it is filtered out by any
    # -R regex that does not name it.
    check(len(registered.failtests) > 50,
          "real tree registers the failtest suite, got %d" % len(registered.failtests))
    sel = select(graph, sorted(registered.failtests)[:1])
    check(sel.mode == "targets" and sel.ctest_names,
          "a changed failtest selects CTest names, got %s (%s)" % (sel.mode, sel.reasons))

    # main.h is a hub: changing it must run everything.
    sel = select(graph, ["test/main.h"])
    check(sel.mode == "all", "test/main.h runs the full suite, got %s" % sel.mode)

    # Private implementation headers may legitimately move.  While present,
    # they must reach their focused test; broadening to the full suite is safe.
    selections = []
    for path, target in (("Eigen/src/Eigenvalues/RealQZ.h", "real_qz"),
                         ("Eigen/src/SVD/BDCSVD.h", "bdcsvd")):
        if path not in graph.files:
            print("skipping real-tree check for absent private header %s" % path)
            continue
        sel = select(graph, [path])
        selections.append(sel)
        check(sel.mode in ("targets", "all"), "%s yields safe coverage, got %s" % (path, sel.mode))
        if sel.mode == "targets":
            check(target in sel.targets, "%s selects %s" % (path, target))

    # Every selected name must come from a real CMake registration.
    registered_targets = set(source_targets.values())
    for sel in selections:
        unknown = sorted(t for t in sel.targets if t not in registered_targets)
        check(not unknown, "selected names are test sources, got %s" % unknown)


def main():
    root = tempfile.mkdtemp(prefix="eigen-affected-")
    try:
        build_fixture(root)
        test_fixture_graph(root)
        test_failtests(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    test_output_encoding()
    test_git_rename_paths()
    test_real_tree()

    if FAILURES:
        print("\n%d check(s) failed" % len(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
