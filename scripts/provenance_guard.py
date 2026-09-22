#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Stop an agent session from looking inside proprietary libraries.

Eigen is compared against vendor BLAS, LAPACK and GPU libraries whose licenses
forbid reverse engineering, and what such inspection reveals must not reach
Eigen (``AGENTS.md`` rule 2, ``.agents/provenance.md``).  This PreToolUse hook
recognizes the mechanical forms of looking inside and answers

  deny   a disassembler, debugger, string or byte dumper whose own arguments
         name a vendor binary, or a file read of one;
  ask    the same tools where the link to a vendor binary is indirect (a
         pipeline, a command substitution), and web requests that pair a vendor
         library with reverse-engineering or leak terms -- the user decides;
  --     nothing for everything else, which includes link diagnosis (``nm``,
         ``ldd``, ``otool -L``, ``objdump -p``), vendor headers, and any
         inspection of Eigen's own or open-source binaries.

It is best effort, a tripwire rather than a proof: a renamed copy of a library
or a custom script passes.  The guidance is the rule; this catches the forms
we could think of.

The hook fails open -- an unparsable payload or an internal error exits 0
without a decision rather than blocking the harness.

Usage:
  provenance_guard.py --claude-hook        # PreToolUse payload on stdin
  provenance_guard.py --command 'CMD'      # exit 0 allow, 1 ask, 2 deny
"""

import json
import re
import shlex
import sys

ALLOW, ASK, DENY = "allow", "ask", "deny"

# License forbids reverse engineering: by file name, then by install root.  Open-source libraries are absent on
# purpose; AOCL's binary packages are listed because their EULA differs from the BSD license of AMD's sources.
_VENDOR_FILE = r"""
    lib(?:mkl|svml|intlc)\w* | lib(?:imf|irc)(?![a-z]) | mkl_\w+\.(?:dll|lib)
  | lib(?:cu(?:blas|solver|sparse|dss|dnn|fft|rand|tensor|da)|nvpl|nvvm|nvrtc|nvjitlink|nvblas)\w*
  | (?:cublas|cusolver|cusparse|cudss|cudnn|cufft|nvrtc)\w*\.dll
  | lib(?:armpl|amath|essl|aocl|amdlibm|sunperf|fjlapack|acml)\w* | lib(?:sci|nag|alm)(?![a-z])
  | (?:Accelerate|vecLib)\.framework | (?-i:lib(?:BLAS|LAPACK|vDSP|BNNS|Sparse)\.dylib) | dyld_shared_cache
"""
_VENDOR_ROOT = r"""
    /opt/intel\b | /oneapi/ | Intel/oneAPI | /opt/nvidia\b | hpc_sdk | /usr/local/cuda | CUDA/v\d
  | /opt/arm\b | arm-performance-libraries | /opt/AMD\b | aocl-linux- | /opt/cray\b | /opt/ibm\b
  | /System/Library/
"""
_VENDOR = re.compile("(?ix)" + _VENDOR_FILE + "|" + _VENDOR_ROOT)
# Internal (undocumented) entry points, as they appear in a symbol filter.
_VENDOR_SYMBOL = r"\bmkl_(?:blas|lapack|spblas|sparse|serv|vml|vsl|pds|dnn)\w*"
# An install root elsewhere in a command is usually a PATH entry or a compiler, so an indirect link needs a name.
_VENDOR_NAMED = re.compile("(?ix)" + _VENDOR_FILE + "|" + _VENDOR_SYMBOL)
_BINARY_EXT = re.compile(r"(?i)\.(?:so|a|dylib|dll|lib|o|obj|fatbin|cubin)(?:\.[\d.]+)?(?!\w)")
# A text file under a vendor root -- a header, a version file, a license -- is documentation, not the box.
_TEXT_EXT = re.compile(r"(?i)\.(?:h|hpp|hh|hxx|inc|inl|cuh|c|cc|cpp|cxx|cu|f|f90|txt|md|rst|json|xml|ya?ml|toml|ini|cfg"
                       r"|cmake|pc|py|sh|csv|log|html?|pdf)$")
_FRAMEWORK_BINARY = re.compile(r"\.framework/(?:[^/]+/)*[^/.]+$")

# tool -> regex over its argument string that says the invocation reads code or
# data out of the file.  "" means every use does.  Flag regexes are
# case-sensitive: `otool -L` and `objdump -T` are link diagnosis.
_SHORT = r"(?:^|\s)-(?!-)[A-Za-z]{0,3}%s[A-Za-z]{0,3}(?=\s|$)"
_DISASSEMBLER = _SHORT % "[dDsS]" + r"|-{1,2}disassemble|-{1,2}full-contents|-{1,2}source\b"
_DEBUGGER = r"disas|x/\d*[a-z]*i\b|layout\s+asm|\b(?:stepi|nexti)\b"
_LOOKS_INSIDE = {
    "objdump": _DISASSEMBLER, "llvm-objdump": _DISASSEMBLER, "gobjdump": _DISASSEMBLER,
    "otool": r"(?:^|\s)-(?!-)[A-Za-z]{0,2}[tsdoj][A-Za-z]{0,2}(?=\s|$)",
    "readelf": _SHORT % "[xw]" + r"|--hex-dump|--debug-dump",
    "dumpbin": r"(?i)/disasm|/rawdata",
    "gdb": _DEBUGGER, "cuda-gdb": _DEBUGGER, "rocgdb": _DEBUGGER, "lldb": _DEBUGGER,
    "perf": r"^\s*annotate\b",
}
for _name in ("strings", "cuobjdump", "nvdisasm", "ndisasm", "r2", "radare2", "rabin2", "rizin", "rz-bin",
              "ghidra", "ghidraRun", "analyzeHeadless", "ida", "ida64", "idat", "idat64",
              "retdec-decompiler", "cstool", "uftrace", "xxd", "hexdump", "hexyl", "od"):
    _LOOKS_INSIDE[_name] = ""
# Text search is looking inside only when it is aimed at the binary itself;
# searching a vendor's headers is ordinary integration work.
_GREP = ("grep", "egrep", "fgrep", "zgrep", "rg", "ag")

_KEYWORDS = ("do", "then", "else", "elif", "if", "while", "until", "!", "{")
# Commands whose arguments are themselves a command.
_RUNNERS = ("ssh", "sudo", "su", "doas", "bash", "sh", "zsh", "dash", "ksh", "eval", "docker", "podman", "wsl",
            "xargs", "parallel", "watch", "script", "find", "env", "timeout", "nohup", "setsid", "nice",
            "ionice", "stdbuf", "taskset", "numactl", "time", "command", "exec", "srun", "mpirun")
_SHELL_FEEDERS = ("ssh", "sudo", "su", "bash", "sh", "zsh", "dash", "ksh", "docker", "podman", "wsl", "env",
                  "timeout", "nohup", "setsid")

_HEREDOC = re.compile(r"""(?<!<)<<(?!<)-?\s*(?:'(\w+)'|"(\w+)"|\\?(\w+))""")
_ASSIGN = re.compile(r"""(?:^|[\s;&|(])([A-Za-z_]\w*)=(\$\([^\n]*\)|"[^"\n]*"|'[^'\n]*'|[^\s;&|)]*)""", re.M)
_ASSIGN_WORD = re.compile(r"^[A-Za-z_]\w*=")
_FOR = re.compile(r"\bfor\s+([A-Za-z_]\w*)\s+in\s+([^;\n]+)")
_VAR = re.compile(r"\$\{?([A-Za-z_]\w*)\}?")
_MAX_DEPTH = 3

_WEB_PRODUCT = re.compile(r"(?i)\b(?:(?:one)?mkl|cublas\w*|cusolver\w*|cusparse\w*|cudss|cudnn|cufft|nvpl|armpl"
                          r"|arm performance librar\w+|accelerate framework|veclib|aocl|essl|libsci)\b")
# A question about the license terms themselves is the diligence this guard exists to encourage.
_WEB_LICENSE = re.compile(r"(?i)licen[sc]e|\bEULA\b|license agreement")
_WEB_INSIDE = re.compile(r"(?i)leak|decompil|disassembl|reverse[- ]?engineer|internal source|confidential"
                         r"|under nda|cracked|\bghidra\b|\bida pro\b")

DENY_MSG = (
    "Eigen provenance guard: `%s` would look inside a proprietary library (%s). Vendor libraries are black boxes "
    "here (AGENTS.md rule 2, .agents/provenance.md).\n"
    "Do not retry through a copy, another tool, or a script. If the target is in fact open source or Eigen's own "
    "binary, tell the user and let them decide."
)
ASK_MSG = (
    "Eigen provenance guard: `%s` may be aimed at a proprietary library (%s appears in the command). Approve only "
    "if the inspected file is Eigen's own or open-source code (.agents/provenance.md)."
)
WEB_MSG = (
    "Eigen provenance guard: this request pairs a proprietary library (%s) with reverse-engineering or leak terms "
    "(%s). Approve it for published documentation, licenses and papers; decline it for leaked source or "
    "third-party disassembly write-ups (.agents/provenance.md)."
)


def vendor_token(text, named_only=False):
    """The first word of `text` naming a vendor library or internal symbol -- or, unless `named_only`, a binary
    under an install root -- else None."""
    for pattern in (_VENDOR_NAMED,) if named_only else (_VENDOR_NAMED, _VENDOR):
        for match in pattern.finditer(text):
            start = max(text.rfind(c, 0, match.start()) for c in " \t\n'\"") + 1
            end = re.compile(r"[\s'\"]|$").search(text, match.end()).start()
            word = text[start:end]
            if not _TEXT_EXT.search(word):
                return word
    return None


def is_vendor_binary(path):
    """True for a path that names a vendor library's binary rather than its headers or documentation."""
    if not _VENDOR.search(path):
        return False
    return bool(_BINARY_EXT.search(path) or _FRAMEWORK_BINARY.search(path))


def _split_heredocs(text):
    """`text` without its heredoc bodies, plus the bodies that are fed to a shell.

    A body written to a file or piped to an interpreter is prose or another
    language, and scanning it for commands would flag a note that merely
    mentions `strings`.
    """
    lines = text.split("\n")
    kept, shell_bodies = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        kept.append(line)
        i += 1
        match = _HEREDOC.search(line)
        if not match:
            continue
        delimiter = match.group(1) or match.group(2) or match.group(3)
        body = []
        while i < len(lines) and lines[i].strip() != delimiter:
            body.append(lines[i])
            i += 1
        i += 1
        consumer = _tokens(re.split(r"[;&|]", line[:match.start()])[-1])
        while consumer and _ASSIGN_WORD.match(consumer[0]):
            consumer.pop(0)
        piped_to_shell = re.search(r"\|\s*(?:ssh|bash|sh|zsh)\b", line[match.end():])
        if (consumer and _base(consumer[0]) in _SHELL_FEEDERS) or piped_to_shell:
            shell_bodies.append("\n".join(body))
    return "\n".join(kept), shell_bodies


def _segments(text):
    """Simple commands of `text`: split at unquoted newlines and `; | & ( )` and backticks."""
    out, buf, quote = [], [], None
    i = 0
    while i < len(text):
        c = text[i]
        if quote:
            buf.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(text):
                i += 1
                buf.append(text[i])
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
            buf.append(c)
        elif c == "\\" and i + 1 < len(text):
            buf.append(c)
            i += 1
            buf.append(text[i])
        elif c in "\n;|&()`":
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    out.append("".join(buf))
    return [segment.strip() for segment in out if segment.strip()]


def _tokens(segment):
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()


def _base(word):
    """The command name of `word`; a relative path is someone's own build product, not a system tool."""
    if "/" in word and not word.startswith("/"):
        return word
    return word.rsplit("/", 1)[-1]


def _collect_env(text):
    env = {}
    for name, value in _ASSIGN.findall(text):
        env.setdefault(name, []).append(value.strip("'\""))
    for name, values in _FOR.findall(text):
        env.setdefault(name, []).append(values)
    return env


def _expand(text, env):
    for _ in range(_MAX_DEPTH):
        expanded = _VAR.sub(lambda m: " ".join(env.get(m.group(1), [m.group(0)])), text)
        if expanded == text:
            break
        text = expanded
    return text


def _grep_files(args, env):
    """File operands of a grep-like command: the positionals after the pattern."""
    files, has_pattern, skip = [], False, False
    for arg in args:
        if skip:
            skip = False
        elif arg in ("-e", "-f", "--regexp", "--file"):
            has_pattern, skip = True, True
        elif arg in ("-A", "-B", "-C", "-m", "-g", "-t", "--include", "--exclude", "--glob", "--type"):
            skip = True
        elif not arg.startswith("-"):
            files.append(arg)
    return _expand(" ".join(files if has_pattern else files[1:]), env).split()


def _lead_index(words):
    """Index of the command word, past shell keywords and `NAME=value` prefixes."""
    i = 0
    while i < len(words) and (words[i] in _KEYWORDS or _ASSIGN_WORD.match(words[i])):
        i += 1
    return i


def _invocation(words):
    """(tool, argument words, runner words) for the inspection tool a simple command runs, else None.

    The runner words are the rest of the command when a runner such as `find`,
    `ssh` or `timeout` leads it, since they can carry the target, and None
    otherwise.  The tool's own path is left out of them: in
    `timeout 60 /usr/local/cuda/bin/cuobjdump -sass bench` the vendor root
    names the tool, not what it inspects.
    """
    i = _lead_index(words)
    if i == len(words):
        return None
    lead = _base(words[i])
    if lead in _LOOKS_INSIDE or lead in _GREP:
        return lead, words[i + 1:], None
    if lead in _RUNNERS:
        for j in range(i + 1, len(words)):
            name = _base(words[j])
            if (name in _LOOKS_INSIDE or name in _GREP) and " " not in words[j]:
                return name, words[j + 1:], words[:j] + words[j + 1:]
    return None


def check_command(command):
    """(verdict, tool, vendor token) for a shell command."""
    text, shell_bodies = _split_heredocs(command.replace("\\\n", " "))
    env = _collect_env("\n".join([text] + shell_bodies))
    unlinked = []

    def visit(script, depth):
        for segment in _segments(script):
            words = _tokens(segment)
            found = _invocation(words)
            lead = _lead_index(words)
            if lead < len(words) and _base(words[lead]) in _RUNNERS and depth < _MAX_DEPTH:
                for word in words:
                    if " " in word or "\n" in word:
                        hit = visit(word, depth + 1)
                        if hit:
                            return hit
            if not found:
                continue
            tool, args, runner_words = found
            arg_text = _expand(" ".join(args), env)
            if tool in _GREP:
                targets = [a for a in _grep_files(args, env) if is_vendor_binary(a)]
                if targets:
                    return DENY, tool, targets[0]
                continue
            if not re.search(_LOOKS_INSIDE[tool], arg_text):
                continue
            # A runner's own arguments can carry the target: `find /opt/intel ... -exec objdump -d {} +`.
            token = vendor_token(_expand(" ".join(runner_words), env) if runner_words else arg_text)
            if token:
                return DENY, tool, token
            unlinked.append(tool)
        return None

    for script in [text] + shell_bodies:
        hit = visit(script, 0)
        if hit:
            return hit
    if unlinked:
        token = vendor_token(_expand("\n".join([text] + shell_bodies), env), named_only=True)
        if token:
            return ASK, unlinked[0], token
    return ALLOW, None, None


def check_web(text):
    """(verdict, product, term) for a web query, URL, or fetch prompt."""
    product, inside = _WEB_PRODUCT.search(text), _WEB_INSIDE.search(text)
    if product and inside and not _WEB_LICENSE.search(text):
        return ASK, product.group(0), inside.group(0)
    return ALLOW, None, None


def check_payload(payload):
    """(verdict, reason) for a PreToolUse payload."""
    tool = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    if tool == "Bash":
        verdict, name, token = check_command(tool_input.get("command") or "")
        if verdict == DENY:
            return DENY, DENY_MSG % (name, token)
        if verdict == ASK:
            return ASK, ASK_MSG % (name, token)
    elif tool in ("Read", "Grep"):
        path = tool_input.get("file_path") or tool_input.get("path") or ""
        if is_vendor_binary(path):
            return DENY, DENY_MSG % (tool, path)
    elif tool in ("WebFetch", "WebSearch"):
        text = " ".join(str(tool_input.get(key) or "") for key in ("query", "url", "prompt"))
        verdict, product, term = check_web(text)
        if verdict == ASK:
            return ASK, WEB_MSG % (product, term)
    return ALLOW, None


def run_hook_mode():
    try:
        verdict, reason = check_payload(json.load(sys.stdin))
    except Exception:
        return 0
    if verdict != ALLOW:
        sys.stdout.write(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": verdict,
            "permissionDecisionReason": reason,
        }}) + "\n")
    return 0


def main():
    if "--claude-hook" in sys.argv[1:]:
        return run_hook_mode()
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--claude-hook", action="store_true", help="run as a Claude Code PreToolUse hook")
    mode.add_argument("--command", metavar="CMD", help="classify one shell command")
    args = parser.parse_args()
    verdict, reason = check_payload({"tool_name": "Bash", "tool_input": {"command": args.command}})
    print(verdict if verdict == ALLOW else "%s: %s" % (verdict, reason))
    return {ALLOW: 0, ASK: 1, DENY: 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
