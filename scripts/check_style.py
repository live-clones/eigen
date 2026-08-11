#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Advisory style check for added C++ code.

Flags the problems that recur in this repository's code reviews: narration-
comment verbosity (see the comment rules in ``AGENTS.md``), and declaration
forms that new code should not use (``NULL``, ``typedef``, ``enum`` constant
blocks, C++17-and-later constructs in the C++14 trees, ``std::`` math in
library headers where ``numext::`` is required).  Only lines a change ADDS
are reported, but each file's complete post-image is lexed so surrounding
context — an enclosing block comment, a Doxygen continuation, a multi-line
initializer — is classified correctly.

The findings are advisory: a flagged construct may be justified, in which
case keep it and state the reason where the construct is.

Modes:
  --diff BASE     check lines added relative to ``merge-base(BASE, HEAD)``,
                  including uncommitted changes and untracked C++ files;
                  exit 1 if findings
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

# Convention checks applied per added code line (comments and literals blanked).
CODE_CHECKS = [
    (r"\bNULL\b", "NULL: new code uses nullptr"),
    (r"^\s*typedef\b", "typedef: prefer `using` in new aliases unless the file is uniformly typedef"),
    (r"std::integral_constant<\s*bool\b", "std::integral_constant<bool,...>: use Eigen's bool_constant (Meta.h)"),
    (r"^\s*enum\s*(:\s*\w+\s*)?\{", "enum constant block: trait/evaluator constants are static constexpr in new "
                                    "code; Flags is `unsigned int`"),
]
CXX14_CHECKS = [
    (r"\bif\s+constexpr\b", "if constexpr: supported code compiles as C++14; use EIGEN_IF_CONSTEXPR(...) with a "
                            "condition that is valid either way"),
    (r"\bstd::(span|optional|variant|string_view|byte)\b", "C++17/20 library type: the supported baseline is C++14"),
]
DESIGNATED_MSG = "designated initializer: C++20-only; use aggregate assignment with /*name=*/ comments"


RAW_PREFIX = re.compile(r"(?:^|[^0-9A-Za-z_])(?:u8|u|U|L)?R$")


def scan_quoted(line, j, quote):
    """Scan a quoted literal's remainder from position ``j``.

    Returns (next_position, closed, spliced): ``spliced`` is True when the
    line ends with a backslash inside the literal, so the literal continues
    on the next physical line.
    """
    n = len(line)
    while j < n:
        if line[j] == "\\":
            if j == n - 1:
                return n, False, True
            j += 2
            continue
        if line[j] == quote:
            return j + 1, True, False
        j += 1
    return n, False, False


def lex_lines(lines):
    """Lex a file's complete contiguous text.

    Returns (code_lines, comment_only) as parallel lists: per line, the code
    with comments and string/character literals blanked, and the stripped
    comment text when the line holds no code (else None).  Block comments,
    raw string literals, and backslash-spliced ordinary literals carry their
    state across physical lines.
    """
    code_lines, comment_only = [], []
    in_block = False
    raw_terminator = None  # e.g. )delim" while inside a raw string literal
    open_quote = None      # quote character of a spliced ordinary literal
    for line in lines:
        out, i, n = [], 0, len(line)
        had_code = False
        literal_at_start = raw_terminator is not None or open_quote is not None
        while i < n:
            if in_block:
                j = line.find("*/", i)
                if j < 0:
                    i = n
                else:
                    in_block = False
                    i = j + 2
                continue
            if raw_terminator is not None:
                j = line.find(raw_terminator, i)
                if j < 0:
                    i = n
                else:
                    i = j + len(raw_terminator)
                    raw_terminator = None
                continue
            if open_quote is not None:
                i, closed, spliced = scan_quoted(line, i, open_quote)
                if closed or not spliced:  # an unspliced open literal is ill-formed; close at EOL
                    open_quote = None
                continue
            c = line[i]
            if c == "/" and i + 1 < n and line[i + 1] == "/":
                break
            if c == "/" and i + 1 < n and line[i + 1] == "*":
                in_block = True
                i += 2
                continue
            if c == '"' and RAW_PREFIX.search(line[:i]):
                paren = line.find("(", i + 1)
                delimiter = line[i + 1:paren] if paren >= 0 else None
                if delimiter is not None and len(delimiter) <= 16 and not re.search(r"[()\\\s]", delimiter):
                    out.append('""')
                    had_code = True
                    raw_terminator = ")" + delimiter + '"'
                    j = line.find(raw_terminator, paren + 1)
                    if j < 0:
                        i = n
                    else:
                        i = j + len(raw_terminator)
                        raw_terminator = None
                    continue
            if c in "\"'":
                out.append(c + c)
                had_code = True
                i, closed, spliced = scan_quoted(line, i + 1, c)
                if not closed and spliced:
                    open_quote = c
                continue
            out.append(c)
            if not c.isspace():
                had_code = True
            i += 1
        stripped = line.strip()
        is_comment = (not had_code) and stripped != "" and not literal_at_start and (
            stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*") or in_block)
        code_lines.append("".join(out))
        comment_only.append(stripped if is_comment else None)
    return code_lines, comment_only


def check_comments(comment_only, added, findings):
    """Flag comment blocks that gain six or more added narration lines.

    Blocks are formed over the full file, so an addition inside an existing
    Doxygen or license block inherits that block's exemption, and only the
    ADDED lines count toward the threshold — extending a pre-existing block
    by a line or two is not reported.
    """
    block_start = None
    for idx in range(len(comment_only) + 1):
        comment = comment_only[idx] if idx < len(comment_only) else None
        if comment is not None:
            if block_start is None:
                block_start = idx
        elif block_start is not None:
            block = comment_only[block_start:idx]
            added_in_block = [line_no for line_no in range(block_start + 1, idx + 1) if line_no in added]
            if added_in_block and len(added_in_block) >= 6 and not (
                    LICENSE.search("\n".join(block)) or DOXYGEN.match(block[0])):
                findings.append((added_in_block[0], "%d added lines of non-Doxygen comment: AGENTS.md keeps only "
                                                    "mathematics, invariants, compatibility constraints, provenance, "
                                                    "or the reason a deliberate construct must not be simplified"
                                                    % len(added_in_block)))
            block_start = None


def check_designated_initializer(code_lines, line_no, findings):
    code = code_lines[line_no - 1]
    hit = re.search(r"[{,]\s*\.\w+\s*=", code)
    if not hit and re.match(r"\s*\.\w+\s*=", code):
        # A line-leading designator continues a braced initializer only when
        # the previous code line ends with `{` or `,`; anything else is a
        # wrapped member access or assignment.
        for prev in range(line_no - 2, -1, -1):
            prev_code = code_lines[prev].rstrip()
            if prev_code:
                hit = prev_code.endswith("{") or prev_code.endswith(",")
                break
    if hit:
        findings.append((line_no, DESIGNATED_MSG))
        return True
    return False


def check_conventions(rel_path, code_lines, added, findings):
    checks = list(CODE_CHECKS)
    if rel_path.startswith(CXX14_TREES):
        checks += CXX14_CHECKS
    if rel_path.startswith(LIBRARY_SRC_TREES):
        checks.append((r"\bstd::(%s)\s*\(" % STD_MATH,
                       "std:: math call in a library header: use the numext:: equivalent "
                       "(device- and custom-scalar-aware)"))
    added_sorted = sorted(added)
    for pattern, message in checks:
        rx = re.compile(pattern)
        for line_no in added_sorted:
            if rx.search(code_lines[line_no - 1]):
                findings.append((line_no, message))
                break  # one report per pattern per file
    for line_no in added_sorted:
        if check_designated_initializer(code_lines, line_no, findings):
            break


def find_findings(rel_path, lines, added):
    """Check one file: ``lines`` is the complete post-image, ``added`` the set
    of 1-based line numbers the change added.  Returns [(line_no, message)]."""
    if os.path.splitext(rel_path)[1].lower() not in CXX_EXT:
        return []
    added = {n for n in added if 1 <= n <= len(lines) and lines[n - 1].strip()}
    if not added:
        return []
    code_lines, comment_only = lex_lines(lines)
    findings = []
    check_comments(comment_only, added, findings)
    check_conventions(rel_path, code_lines, added, findings)
    findings.sort()
    return findings


def added_lines_from_diff(diff_text):
    """Parse ``git diff -U0`` output into {relative_path: set(added line numbers)}."""
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
                files.setdefault(path, set()).add(new_line)
            new_line += 1
        elif not raw.startswith("-"):
            new_line += 1
    return files


def git(root, *args):
    return subprocess.run(["git"] + list(args), cwd=root, capture_output=True, text=True)


def read_post_image(root, rel_path):
    try:
        with open(os.path.join(root, rel_path), encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return None


def run_diff_mode(base, root=REPO_ROOT):
    """Return [(rel_path, line_no, message)] for lines added since
    merge-base(base, HEAD), including untracked C++ files."""
    merge_base = git(root, "merge-base", base, "HEAD")
    resolved = merge_base.stdout.strip() if merge_base.returncode == 0 else base
    diff = git(root, "diff", "-U0", "--no-color", resolved)
    if diff.returncode not in (0, 1):
        sys.stderr.write(diff.stderr)
        raise SystemExit(2)
    per_file = added_lines_from_diff(diff.stdout)
    untracked = git(root, "ls-files", "--others", "--exclude-standard")
    for rel_path in untracked.stdout.splitlines():
        if os.path.splitext(rel_path)[1].lower() in CXX_EXT:
            text = read_post_image(root, rel_path)
            if text is not None:
                per_file.setdefault(rel_path, set()).update(range(1, len(text.splitlines()) + 1))
    results = []
    for rel_path in sorted(per_file):
        text = read_post_image(root, rel_path)
        if text is None:  # deleted or unreadable
            continue
        for line_no, message in find_findings(rel_path, text.splitlines(), per_file[rel_path]):
            results.append((rel_path, line_no, message))
    return results


def added_from_structured_patch(tool_response):
    """Derive added line numbers from the tool response's structured patch,
    which records the exact edited hunks.  Returns None when absent or of an
    unexpected shape, so callers fall through to the heuristics."""
    try:
        added = set()
        for hunk in tool_response["structuredPatch"]:
            line_no = int(hunk["newStart"])
            for entry in hunk["lines"]:
                if entry.startswith("+"):
                    added.add(line_no)
                    line_no += 1
                elif not entry.startswith("-"):
                    line_no += 1
        return added or None
    except Exception:
        return None


def hook_added_line_numbers(content, snippets):
    """Map the strings an edit added onto post-image line numbers.

    Each snippet must occur exactly once — a repeated occurrence cannot be
    told apart from pre-existing identical text, and would mark unrelated
    lines.  A snippet's span excludes the line after its trailing newline.
    Returns None when any snippet is absent or ambiguous; the caller then
    checks the snippet text standalone."""
    added = set()
    for snippet in snippets:
        if not snippet:
            continue
        start = content.find(snippet)
        if start < 0 or content.find(snippet, start + 1) >= 0:
            return None
        first = content.count("\n", 0, start) + 1
        span = snippet.count("\n") + (0 if snippet.endswith("\n") else 1)
        added.update(range(first, first + max(span, 1)))
    return added


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
        snippets = [tool_input.get("content", "")]
    elif tool_name == "Edit":
        snippets = [tool_input.get("new_string", "")]
    elif tool_name == "MultiEdit":
        snippets = [e.get("new_string", "") for e in tool_input.get("edits", [])]
    else:
        return 0
    if not any(s.strip() for s in snippets):
        return 0

    # The hook runs after the edit, so the file on disk is the post-image;
    # lex it whole so enclosing comments and initializers classify correctly.
    # The edited lines come from the response's structured patch when present,
    # else from locating a uniquely occurring added string.
    text = read_post_image(REPO_ROOT, rel_path)
    added = None
    if text is not None:
        lines = text.splitlines()
        added = added_from_structured_patch(payload.get("tool_response") or {})
        if added is None:
            if tool_name == "Write":
                added = set(range(1, len(lines) + 1))
            else:
                added = hook_added_line_numbers(text, snippets)
    if added is None:  # unreadable file or unlocatable snippet: check the added text standalone
        lines = "\n".join(snippets).splitlines()
        added = set(range(1, len(lines) + 1))
    findings = find_findings(rel_path, lines, added)
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
    mode.add_argument("--diff", metavar="BASE", help="check lines added relative to merge-base(BASE, HEAD)")
    mode.add_argument("--claude-hook", action="store_true", help="run as a Claude Code PostToolUse hook")
    args = parser.parse_args()
    if args.claude_hook:
        try:
            return run_hook_mode()
        except Exception:
            return 0  # a broken hook must not block the harness
    results = run_diff_mode(args.diff)
    for rel_path, line_no, message in results:
        print("%s:%d: %s" % (rel_path, line_no, message))
    if results:
        print("\n%d advisory finding(s) in added lines. Keep a flagged construct only if it is justified, "
              "and say why at the construct." % len(results))
    return 1 if results else 0


if __name__ == "__main__":
    sys.exit(main())
