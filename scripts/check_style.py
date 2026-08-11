#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Advisory style check for added C++ code.

Flags the problems that recur in this repository's code reviews and that
``AGENTS.md`` / ``.agents/conventions.md`` describe: narration-comment
verbosity, and declaration forms that new code should not use (``NULL``,
``typedef``, ``enum`` constant blocks, C++17-and-later constructs in the
C++14 trees, ``std::`` math in library headers).  Only lines a change ADDS
are inspected, so pre-existing code is never reported.

The findings are advisory: a flagged construct may be justified, in which
case keep it and say why where the construct is.

Modes:
  --diff BASE     check lines added relative to git revision BASE
                  (uncommitted changes included); exit 1 if findings
  --claude-hook   run as a Claude Code PostToolUse hook: read the tool-call
                  JSON from stdin and check the text the edit added; exit 2
                  if findings so the harness feeds them back to the model

The Claude Code hook is registered for this repository in
``.claude/settings.json``.  Other agents and humans can run the diff mode
directly, e.g. ``python3 scripts/check_style.py --diff origin/master``.
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CXX_EXT = {".h", ".hpp", ".hxx", ".cpp", ".cc", ".cxx", ".cu", ".cuh", ".inc"}

# Trees whose headers and tests must compile as C++14 (see AGENTS.md rule 3).
CXX14_TREES = ("Eigen/", "unsupported/Eigen/", "test/", "unsupported/test/", "failtest/", "blas/", "lapack/")
# Library implementation headers, where numext:: is required over std:: math.
LIBRARY_SRC_TREES = ("Eigen/src/", "unsupported/Eigen/src/")

DOXYGEN = re.compile(r"^\s*(/\*\*|/\*!|///|//!)")
LICENSE = re.compile(r"SPDX|Copyright|License|Mozilla Public", re.I)
STD_MATH = (
    r"abs|sqrt|isnan|isinf|isfinite|exp|expm1|log|log1p|log2|pow|sin|cos|tan|asin|acos|atan|atan2|"
    r"sinh|cosh|tanh|hypot|fma|ldexp|frexp|floor|ceil|round|rint|trunc|fmod"
)

# Convention checks applied to added code (comments and string literals blanked).
# (regex, flags, message)
CODE_CHECKS = [
    (r"\bNULL\b", 0, "NULL: new code uses nullptr (.agents/conventions.md)"),
    (r"^\s*typedef\b", re.M, "typedef: prefer `using` in new aliases unless the file is uniformly typedef "
                             "(.agents/conventions.md)"),
    (r"std::integral_constant<\s*bool\b", 0, "std::integral_constant<bool,...>: use Eigen's bool_constant (Meta.h)"),
    (r"^\s*enum\s*(:\s*\w+\s*)?\{", re.M, "enum constant block: trait/evaluator constants are static constexpr in "
                                          "new code; Flags is `unsigned int` (.agents/conventions.md)"),
    (r"[{,]\s*\.\w+\s*=", 0, "designated initializer: C++20-only; use aggregate assignment with /*name=*/ comments"),
]
CXX14_CHECKS = [
    (r"\bif\s+constexpr\b", 0, "if constexpr: supported code compiles as C++14; use EIGEN_IF_CONSTEXPR(...) with a "
                               "condition that is valid either way"),
    (r"\bstd::(span|optional|variant|string_view|byte)\b", 0, "C++17/20 library type: the supported baseline is "
                                                              "C++14 (.agents/conventions.md)"),
]


def blank_comments_and_strings(numbered_lines):
    """Blank comments and string/character literals from added lines.

    ``numbered_lines`` is a list of (line_number, text) for the ADDED lines
    only.  Returns (code_lines, comment_only) as parallel lists: the code with
    comments/literals blanked, and the stripped comment text when a line is
    comment-only (else None).  Block-comment state resets across
    non-contiguous line numbers, since the intervening lines are unknown.
    """
    code_lines, comment_only = [], []
    in_block = False
    prev_no = None
    for line_no, line in numbered_lines:
        if prev_no is not None and line_no != prev_no + 1:
            in_block = False
        prev_no = line_no
        out, i, n = [], 0, len(line)
        had_code = False
        while i < n:
            if in_block:
                j = line.find("*/", i)
                if j < 0:
                    i = n
                else:
                    in_block = False
                    i = j + 2
                continue
            c = line[i]
            if c == "/" and i + 1 < n and line[i + 1] == "/":
                break
            if c == "/" and i + 1 < n and line[i + 1] == "*":
                in_block = True
                i += 2
                continue
            if c in "\"'":
                quote, j = c, i + 1
                while j < n:
                    if line[j] == "\\":
                        j += 2
                        continue
                    if line[j] == quote:
                        break
                    j += 1
                out.append(quote + quote)
                i = j + 1 if j < n else n
                had_code = True
                continue
            out.append(c)
            if not c.isspace():
                had_code = True
            i += 1
        stripped = line.strip()
        is_comment = (not had_code) and stripped != "" and (
            stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*") or in_block)
        code_lines.append("".join(out))
        comment_only.append(stripped if is_comment else None)
    return code_lines, comment_only


def check_comments(numbered_lines, comment_only, findings):
    """Flag long narration blocks.

    Doxygen blocks and license headers are exempt; the threshold is
    deliberately loose so justified mathematics or invariant comments are
    rarely touched.
    """
    blocks, block = [], []
    prev_no = None
    for (line_no, _), comment in zip(numbered_lines, comment_only):
        contiguous = prev_no is not None and line_no == prev_no + 1
        if comment is not None:
            if block and not contiguous:
                blocks.append(block)
                block = []
            block.append((line_no, comment))
        else:
            if block:
                blocks.append(block)
            block = []
        prev_no = line_no
    if block:
        blocks.append(block)

    for blk in blocks:
        text = "\n".join(c for _, c in blk)
        if LICENSE.search(text) or DOXYGEN.match(blk[0][1]):
            continue
        if len(blk) >= 6:
            findings.append((blk[0][0], "%d-line non-Doxygen comment block: AGENTS.md keeps only mathematics, "
                                        "invariants, compatibility constraints, provenance, or the reason a "
                                        "deliberate construct must not be simplified" % len(blk)))


def check_conventions(rel_path, numbered_lines, code_lines, findings):
    checks = list(CODE_CHECKS)
    if rel_path.startswith(CXX14_TREES):
        checks += CXX14_CHECKS
    if rel_path.startswith(LIBRARY_SRC_TREES):
        checks.append((r"\bstd::(%s)\s*\(" % STD_MATH, 0,
                       "std:: math call in a library header: use the numext:: equivalent "
                       "(device- and custom-scalar-aware)"))
    for pattern, flags, message in checks:
        rx = re.compile(pattern, flags)
        for (line_no, _), code in zip(numbered_lines, code_lines):
            if rx.search(code):
                findings.append((line_no, message))
                break  # one report per pattern per file


def check_added_lines(rel_path, numbered_lines):
    """Check the added lines of one file; returns [(line_number, message)]."""
    if os.path.splitext(rel_path)[1].lower() not in CXX_EXT:
        return []
    numbered_lines = [(n, l) for n, l in numbered_lines]
    if not any(l.strip() for _, l in numbered_lines):
        return []
    code_lines, comment_only = blank_comments_and_strings(numbered_lines)
    findings = []
    check_comments(numbered_lines, comment_only, findings)
    check_conventions(rel_path, numbered_lines, code_lines, findings)
    findings.sort()
    return findings


def added_lines_from_diff(diff_text):
    """Parse ``git diff -U0`` output into {relative_path: [(line_no, text)]}."""
    files = {}
    path, new_line = None, 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            name = raw[4:]
            path = None if name == "/dev/null" else name[2:] if name.startswith("b/") else name
        elif raw.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,(\d+))?", raw)
            new_line = int(m.group(1)) if m else 0
        elif raw.startswith("+") and not raw.startswith("+++"):
            if path is not None:
                files.setdefault(path, []).append((new_line, raw[1:]))
            new_line += 1
        elif not raw.startswith("-"):
            new_line += 1
    return files


def run_diff_mode(base):
    diff = subprocess.run(["git", "diff", "-U0", "--no-color", base], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    if diff.returncode not in (0, 1):
        sys.stderr.write(diff.stderr)
        return 2
    total = 0
    for rel_path, lines in sorted(added_lines_from_diff(diff.stdout).items()):
        for line_no, message in check_added_lines(rel_path, lines):
            print("%s:%d: %s" % (rel_path, line_no, message))
            total += 1
    if total:
        print("\n%d advisory finding(s) in added lines. Keep a flagged construct only if it is justified, "
              "and say why at the construct." % total)
    return 1 if total else 0


def run_hook_mode():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    tool_input = payload.get("tool_input", {}) or {}
    tool_name = payload.get("tool_name", "")
    path = tool_input.get("file_path", "") or ""
    rel_path = os.path.relpath(os.path.abspath(path), REPO_ROOT).replace(os.sep, "/")
    if rel_path.startswith(".."):
        return 0
    if tool_name == "Write":
        added = tool_input.get("content", "")
    elif tool_name == "Edit":
        added = tool_input.get("new_string", "")
    elif tool_name == "MultiEdit":
        added = "\n".join(e.get("new_string", "") for e in tool_input.get("edits", []))
    else:
        return 0
    findings = check_added_lines(rel_path, list(enumerate(added.splitlines(), start=1)))
    if findings:
        sys.stderr.write("style check (%s) — review before proceeding:\n" % rel_path)
        for _, message in findings[:8]:
            sys.stderr.write("  - %s\n" % message)
        sys.stderr.write("Advisory: keep a flagged construct only if it is justified, and say why at the "
                         "construct.\n")
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diff", metavar="BASE", help="check lines added relative to git revision BASE")
    mode.add_argument("--claude-hook", action="store_true", help="run as a Claude Code PostToolUse hook")
    args = parser.parse_args()
    if args.claude_hook:
        try:
            return run_hook_mode()
        except Exception:
            return 0  # a broken hook must not block the harness
    return run_diff_mode(args.diff)


if __name__ == "__main__":
    sys.exit(main())
