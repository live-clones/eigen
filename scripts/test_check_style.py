#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for scripts/check_style.py.

Exercises the checks on synthetic added text and the diff parser on a crafted
diff, so the expectations do not depend on the checked-out tree.

Usage: python3 scripts/test_check_style.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_style import added_lines_from_diff, check_added_lines


def numbered(text, start=1):
    return list(enumerate(text.splitlines(), start=start))


def messages(rel_path, text):
    return [m for _, m in check_added_lines(rel_path, numbered(text))]


def assert_flags(rel_path, text, *fragments):
    got = messages(rel_path, text)
    for fragment in fragments:
        assert any(fragment in m for m in got), "expected %r in findings for %s, got %r" % (fragment, rel_path, got)


def assert_clean(rel_path, text):
    got = messages(rel_path, text)
    assert not got, "expected no findings for %s, got %r" % (rel_path, got)


def test_conventions_flagged():
    assert_flags("Eigen/src/Core/Foo.h", "const char* p = NULL;\n", "NULL")
    assert_flags("Eigen/src/Core/Foo.h", "typedef int MyInt;\n", "typedef")
    assert_flags("Eigen/src/Core/Foo.h", "std::integral_constant<bool, true> b;\n", "bool_constant")
    assert_flags("Eigen/src/Core/Foo.h", "enum { Flags = 0 };\n", "static constexpr")
    assert_flags("Eigen/src/Core/Foo.h", "S s = {.size = 3};\n", "designated initializer")
    assert_flags("Eigen/src/Core/Foo.h", "if constexpr (kSize > 4) {}\n", "EIGEN_IF_CONSTEXPR")
    assert_flags("test/foo.cpp", "std::optional<int> x;\n", "C++17/20 library type")
    assert_flags("Eigen/src/Core/Foo.h", "double x = std::sqrt(2.0);\n", "numext::")


def test_scoping():
    # std:: math is only flagged in library implementation headers.
    assert_clean("test/foo.cpp", "double x = std::sqrt(2.0);\n")
    # C++14 checks do not apply outside the C++14 trees.
    assert_clean("benchmarks/Core/foo.cpp", "if constexpr (kSize > 4) {}\n")
    # Non-C++ files are ignored entirely.
    assert_clean("AGENTS.md", "NULL typedef enum {\n")


def test_false_positive_probes():
    assert_clean("Eigen/src/Core/Foo.h", "opts.size = 4;\nfoo(a.b, c.d);\n")           # member access, not init
    assert_clean("Eigen/src/Core/Foo.h", "double v[] = {.5, 1.5};\n")                  # float literal, not init
    assert_clean("Eigen/src/Core/Foo.h", "enum class Kind { A, B };\n")                # scoped enums are fine
    assert_clean("Eigen/src/Core/Foo.h", 'const char* s = "NULL typedef enum {";\n')   # inside a string
    assert_clean("Eigen/src/Core/Foo.h", "// NULL and typedef discussed in a comment\nint x = 1;\n")
    assert_clean("Eigen/src/Core/Foo.h", "using MyInt = int;\nstatic constexpr unsigned int Flags = 0;\n"
                                         "const char* p = nullptr;\nEIGEN_IF_CONSTEXPR (kSize > 4) {}\n"
                                         "double x = numext::sqrt(2.0);\n")


def test_comment_verbosity():
    narration = "\n".join("// narration line %d" % i for i in range(6)) + "\nint x = 1;\n"
    assert_flags("Eigen/src/Core/Foo.h", narration, "non-Doxygen comment block")
    # License headers and Doxygen blocks are exempt however long they are.
    license_header = "\n".join("// SPDX-License-Identifier: MPL-2.0" if i == 0 else "// Copyright notice %d" % i
                               for i in range(8)) + "\nint x = 1;\n"
    assert_clean("Eigen/src/Core/Foo.h", license_header)
    doxygen = "/** \\brief Documented API.\n" + "\n".join(" * line %d" % i for i in range(8)) + "\n */\nint x = 1;\n"
    assert_clean("Eigen/src/Core/Foo.h", doxygen)
    # Five lines stay under the block threshold.
    assert_clean("Eigen/src/Core/Foo.h", "\n".join("// l%d" % i for i in range(5)) + "\nint x = 1;\n")
    # Non-contiguous added comment lines (as in a diff) do not merge into one block.
    scattered = [(1, "// a"), (2, "// b"), (3, "// c"), (10, "// d"), (11, "// e"), (12, "// f"), (13, "int x;")]
    assert not check_added_lines("Eigen/src/Core/Foo.h", scattered)


def test_block_comment_state():
    # A /* ... */ block spanning added lines hides code checks inside it...
    assert_clean("Eigen/src/Core/Foo.h", "/* start\n NULL typedef\n end */\nint x = 1;\n")
    # ...and block state resets across a gap in line numbers.
    gap = [(1, "/* start"), (50, "const char* p = NULL;")]
    assert any("NULL" in m for _, m in check_added_lines("Eigen/src/Core/Foo.h", gap))


def test_diff_parser():
    diff = (
        "diff --git a/Eigen/src/Core/Foo.h b/Eigen/src/Core/Foo.h\n"
        "--- a/Eigen/src/Core/Foo.h\n"
        "+++ b/Eigen/src/Core/Foo.h\n"
        "@@ -10,0 +11,2 @@ context\n"
        "+const char* p = NULL;\n"
        "+int y = 2;\n"
        "@@ -20,1 +23,1 @@ context\n"
        "-old line\n"
        "+typedef int T;\n"
        "diff --git a/gone.cpp b/gone.cpp\n"
        "--- a/gone.cpp\n"
        "+++ /dev/null\n"
    )
    files = added_lines_from_diff(diff)
    assert set(files) == {"Eigen/src/Core/Foo.h"}, files
    assert files["Eigen/src/Core/Foo.h"] == [(11, "const char* p = NULL;"), (12, "int y = 2;"),
                                             (23, "typedef int T;")], files
    findings = check_added_lines("Eigen/src/Core/Foo.h", files["Eigen/src/Core/Foo.h"])
    lines = sorted(line for line, _ in findings)
    assert lines == [11, 23], findings


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS %s" % test.__name__)
    print("%d tests passed" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
