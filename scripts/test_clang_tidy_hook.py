#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for scripts/clang_tidy_hook.py and scripts/style_common.py.

The line-filter construction, umbrella resolution and target selection are
tested against the real tree without invoking clang-tidy.  The end-to-end
check does invoke it, and is skipped when clang-tidy is not installed.

Usage: python3 scripts/test_clang_tidy_hook.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clang_tidy_hook import compile_args, module_of, tidy_target, umbrella_for
from style_common import REPO_ROOT, as_ranges, line_filter_json


def test_as_ranges():
    assert as_ranges(set()) == []
    assert as_ranges({5}) == [[5, 5]]
    # Consecutive numbers collapse; gaps start a new range.
    assert as_ranges({1, 2, 3}) == [[1, 3]]
    assert as_ranges({3, 1, 2, 7, 8, 20}) == [[1, 3], [7, 8], [20, 20]]


def test_line_filter_json():
    payload = json.loads(line_filter_json({"Eigen/src/Core/Block.h": {10, 11, 40}}))
    assert payload == [{"name": "Eigen/src/Core/Block.h", "lines": [[10, 11], [40, 40]]}], payload
    # Repository-relative paths, not basenames: `Block.h` alone would match any
    # module's Block.h, since clang-tidy compares the name as a path suffix.
    assert "/" in payload[0]["name"]
    # A file with no added lines must not appear: an empty entry would be a
    # filter that silences the file rather than one that reports nothing.
    assert json.loads(line_filter_json({"a.h": set(), "b.h": {1}})) == [{"name": "b.h", "lines": [[1, 1]]}]


def test_module_of():
    assert module_of("Eigen/src/Core/Block.h") == "Core"
    assert module_of("Eigen/src/Core/arch/AVX/PacketMath.h") == "Core"
    assert module_of("unsupported/Eigen/src/Tensor/TensorBlock.h") == "Tensor"
    assert module_of("test/block.cpp") is None
    assert module_of("Eigen/Core") is None


def test_umbrella_resolution():
    # Read from the module's InternalHeaderCheck.h `#error` directive.
    assert umbrella_for("Eigen/src/Core/Block.h") == "Eigen/Core"
    # The Tensor module's umbrella lives outside its own directory tree.
    assert umbrella_for("unsupported/Eigen/src/Tensor/TensorBlock.h") == "unsupported/Eigen/Tensor"
    # Arch backends nested below the module root carry no directive of their
    # own and fall back to <root>/<Module>.
    assert umbrella_for("Eigen/src/Core/arch/AVX/PacketMath.h") == "Eigen/Core"
    # Modules needing a third-party header are skipped rather than guessed at.
    assert umbrella_for("Eigen/src/CholmodSupport/CholmodSupport.h") is None
    # Every module in the tree must resolve, or the hook silently skips files.
    unresolved = []
    for tree in ("Eigen/src", "unsupported/Eigen/src"):
        for module in sorted(os.listdir(os.path.join(REPO_ROOT, tree))):
            rel = "%s/%s/InternalHeaderCheck.h" % (tree, module)
            if not os.path.isfile(os.path.join(REPO_ROOT, rel)):
                continue
            if umbrella_for(rel) is None and module not in (
                    "AccelerateSupport", "CholmodSupport", "KLUSupport", "MetisSupport",
                    "PaStiXSupport", "PardisoSupport", "SPQRSupport", "SuperLUSupport",
                    "UmfPackSupport"):
                unresolved.append(rel)
    assert not unresolved, unresolved


def test_tidy_target():
    tmp = tempfile.mkdtemp(prefix="tidy_target_test_")
    try:
        # A src-tree header is linted through a generated umbrella driver.
        driver = tidy_target("Eigen/src/Core/Block.h", tmp)
        assert driver is not None and driver.startswith(tmp)
        assert open(driver).read() == "#include <Eigen/Core>\n"
        # An extensionless public module header is included directly.
        driver = tidy_target("Eigen/Core", tmp)
        assert open(driver).read() == "#include <Eigen/Core>\n"
        # A test source is linted in place, not through a driver.
        assert tidy_target("test/block.cpp", tmp) == os.path.join(REPO_ROOT, "test/block.cpp")
        # A header outside the src trees has no reliable standalone TU.
        assert tidy_target("test/main.h", tmp) is None
        # An unresolvable module is skipped.
        assert tidy_target("Eigen/src/CholmodSupport/CholmodSupport.h", tmp) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_compile_args():
    args = compile_args("Eigen/src/Core/Block.h")
    assert "-std=c++14" in args and "-I" + REPO_ROOT in args
    # Test sources include main.h from test/.
    assert any(a.endswith("/test") for a in compile_args("test/block.cpp"))
    assert any(a.endswith("/test") for a in compile_args("unsupported/test/cxx11_tensor_block_access.cpp"))
    assert not any(a.endswith("/test") for a in compile_args("Eigen/src/Core/Block.h"))


def test_end_to_end_line_filter():
    """clang-tidy must report an added NULL and stay silent on legacy ones."""
    if shutil.which("clang-tidy") is None:
        print("SKIP test_end_to_end_line_filter (clang-tidy not installed)")
        return
    tmp = tempfile.mkdtemp(prefix="tidy_e2e_test_")
    try:
        header = os.path.join(tmp, "probe.h")
        with open(header, "w") as handle:
            handle.write("struct Probe {\n"
                         "  typedef int Legacy;\n"      # line 2: pre-existing
                         "  typedef int Added;\n"       # line 3: "added"
                         "};\n")
        source = os.path.join(tmp, "probe.cpp")
        with open(source, "w") as handle:
            handle.write('#include "probe.h"\n')
        cmd = ["clang-tidy", "--quiet", "--checks=-*,modernize-use-using",
               "--header-filter=probe\\.h",
               "--line-filter=" + json.dumps([{"name": "probe.h", "lines": [[3, 3]]}]),
               source, "--", "-std=c++14", "-I" + tmp]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        hits = [line for line in out.splitlines() if "modernize-use-using" in line]
        assert len(hits) == 1, "expected exactly the added line, got %r" % (hits,)
        assert "probe.h:3:" in hits[0], hits
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_broken_translation_unit_is_not_reported_clean():
    """A file that does not compile must be named, not silently passed."""
    if shutil.which("clang-tidy") is None:
        print("SKIP test_broken_translation_unit_is_not_reported_clean (clang-tidy not installed)")
        return
    tmp = tempfile.mkdtemp(prefix="tidy_broken_test_")
    try:
        os.makedirs(os.path.join(tmp, "test"))
        # run_clang_tidy reads <root>/.clang-tidy, so the fake root needs one.
        shutil.copyfile(os.path.join(REPO_ROOT, ".clang-tidy"), os.path.join(tmp, ".clang-tidy"))
        source = os.path.join(tmp, "test", "broken.cpp")
        with open(source, "w") as handle:
            handle.write("int valid = 0;\nthis is not valid C++ at all @@@;\n")
        from clang_tidy_hook import run_clang_tidy
        diagnostics, skipped = run_clang_tidy({"test/broken.cpp": {2}}, root=tmp)
        assert diagnostics == [], diagnostics
        reasons = [reason for path, reason in skipped if path == "test/broken.cpp"]
        assert reasons == ["translation unit did not compile"], skipped
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS %s" % test.__name__)
    print("%d tests passed" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
