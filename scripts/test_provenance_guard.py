#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for scripts/provenance_guard.py.

The deny cases include the command shapes that prompted the guard; the allow
cases are the neighboring legitimate work it must not interrupt -- link
diagnosis, Eigen's own assembly analysis, and prose that mentions the tools.

Usage: python3 scripts/test_provenance_guard.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from provenance_guard import ALLOW, ASK, DENY, check_command, check_payload, check_web, is_vendor_binary

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "provenance_guard.py")


def verdict(command):
    return check_command(command)[0]


def payload(tool, **tool_input):
    return {"tool_name": tool, "tool_input": tool_input}


def test_direct_inspection_is_denied():
    assert verdict("strings -a /opt/arm/armpl_26.07_flang-22/lib/libarmpl_lp64.dylib") == DENY
    assert verdict("sudo objdump -D /usr/lib/x86_64-linux-gnu/libcudss.so.0 | head") == DENY
    assert verdict("llvm-objdump --disassemble -C /opt/intel/oneapi/mkl/latest/lib/libmkl_core.so") == DENY
    assert verdict("otool -tV /System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate") == DENY
    assert verdict("otool -tv /System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Versions/A/"
                   "libBLAS.dylib") == DENY
    assert verdict("sudo strings -n 8 /opt/arm/lib/libarmpl_lp64.dylib") == DENY
    assert verdict("cuobjdump -xelf all /usr/local/cuda/lib64/libcublas.so.12") == DENY
    assert verdict("xxd /usr/local/cuda-13/lib64/libcusolver.so | head -50") == DENY
    assert verdict("perf annotate --stdio -d libmkl_avx2.so.2") == DENY
    assert verdict("gdb -batch -ex 'disas mkl_blas_example_kernel' ./bench_gemm") == DENY
    assert verdict("grep -a -c SWITCH_ /opt/arm/lib/libarmpl_lp64.dylib") == DENY
    assert verdict("rg -a -e FOO -e BAR libarmpl_lp64.dylib") == DENY


def test_variables_loops_and_runners_are_followed():
    assert verdict('LIB=$(ls /usr/local/cuda/nvvm/lib64/libnvvm.so* 2>/dev/null | head -1); echo "$LIB"\n'
                   '[ -n "$LIB" ] && strings "$LIB" | grep -i some-flag') == DENY
    assert verdict("for f in /opt/intel/oneapi/mkl/latest/lib/*.so; do objdump -d ${f} | wc -l; done") == DENY
    assert verdict("find /usr/local/cuda/lib64 -name 'libcublas*' -exec objdump -d {} +") == DENY
    assert verdict("bash -c 'strings /opt/arm/lib/libarmpl_lp64.dylib | grep SWITCH_'") == DENY
    assert verdict("LC_ALL=C bash -c 'objdump -d /opt/intel/oneapi/mkl/latest/lib/libmkl_core.so'") == DENY
    assert verdict("for h in a b; do ssh $h 'strings /opt/arm/lib/libarmpl_lp64.dylib'; done") == DENY
    assert verdict("ssh -n -p 22 user@host 'for L in /usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0 "
                   "/opt/AMD/aocl/aocl-linux-aocc-5.2.0/aocc/lib_LP64/libblis-mt.so.5.2.0; do echo \"== $L\"; "
                   "objdump -d --no-show-raw-insn -C $L | grep -A30 \"<ddot_>:\"; done'") == DENY
    assert verdict("ssh -o BatchMode=yes user@host 'bash -s' <<'EOF' 2>&1\n"
                   "L=/opt/arm/armpl_26.07_flang-22/lib/libarmpl_lp64.dylib\n"
                   "otool -tv $L | grep -c fmopa\n"
                   "EOF") == DENY


def test_indirect_links_ask():
    assert verdict("find /opt/arm -name 'libarmpl*' | xargs strings") == ASK
    assert verdict("objdump -d $(find /opt/intel -name libmkl_core.so)") == ASK
    assert verdict("objdump -d build/bench_gemm | grep -A20 mkl_blas_example_kernel") == ASK


def test_link_diagnosis_is_allowed():
    assert verdict("nm -D --defined-only /opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/lib/libpgmath.so") == ALLOW
    assert verdict("otool -L /opt/arm/armpl_26.07_flang-22/lib/libarmpl_lp64.dylib") == ALLOW
    assert verdict("objdump -p /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so | grep NEEDED") == ALLOW
    assert verdict("readelf -d /usr/lib/x86_64-linux-gnu/libcudss.so.0") == ALLOW
    assert verdict("ldd build/bench_gemm | grep libmkl") == ALLOW
    assert verdict("grep -rn cblas_dgemm /opt/intel/oneapi/mkl/latest/include") == ALLOW
    # Text files under a vendor root are documentation, whatever tool reads them.
    assert verdict("xxd /usr/local/cuda/include/cublas_v2.h | head") == ALLOW
    assert verdict("od -c /opt/intel/oneapi/mkl/latest/include/mkl.h | head -3") == ALLOW
    assert verdict("hexdump -C /usr/local/cuda/version.json") == ALLOW
    # Tracing calls into the library's public API observes it from outside.
    assert verdict("ltrace -l libmkl_rt.so ./build/bench_gemm") == ALLOW
    # A library name as the pattern, not the file searched.
    assert verdict("ls /opt/arm/lib | grep -E '^libarmpl(_lp64)?\\.dylib$'") == ALLOW
    assert verdict("grep -v libcudart_static.a build/build.ninja") == ALLOW


def test_own_and_open_source_binaries_are_allowed():
    assert verdict("objdump -d --no-show-raw-insn -C build/benchmarks/bench_dot | less") == ALLOW
    # A build directory named after the library it links is still Eigen's binary.
    assert verdict("B=~/eigen/wt/build-comparison; otool -tv $B/aarch64-neon__armpl_neon/comparison/bench_gemm_compare "
                   "| grep -c fmopa") == ALLOW
    assert verdict("gdb -batch -ex run -ex 'bt 15' --args ./build/unsupported/test/GPU/cusparse_spmv_4") == ALLOW
    assert verdict("ssh user@host 'bash -s' <<'EOF'\nL=/opt/homebrew/opt/openblas/lib/libopenblas.dylib\n"
                   "otool -tv $L | grep -c -i fmopa\nEOF") == ALLOW
    assert verdict("perf annotate --stdio -s Eigen::internal::gebp_kernel") == ALLOW
    # A vendor install root on PATH does not make Eigen's binary a vendor one.
    assert verdict("PATH=/usr/local/cuda-13.3/bin:$PATH; nvcc -o t t.cu && cuobjdump -ptx ./t; strings ./t") == ALLOW
    # Nor does the tool's own install path, with or without a runner in front.
    assert verdict("/usr/local/cuda/bin/cuobjdump -sass build/test/gpu_basic") == ALLOW
    assert verdict("timeout 60 /usr/local/cuda/bin/cuobjdump -sass build/test/gpu_basic") == ALLOW
    assert verdict("env CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/cuobjdump -sass build/test/gpu_basic") == ALLOW
    # Homebrew's reference LAPACK is open source; only Apple's capitalized names are Accelerate.
    assert verdict("otool -tv /opt/homebrew/opt/lapack/lib/liblapack.dylib | head") == ALLOW
    # `./r2` is a reproducer, not radare2.
    assert verdict("gfortran repro2.f90 -o r2 -L /opt/intel/oneapi/mkl/latest/lib -lmkl_rt && ./r2") == ALLOW


def test_prose_is_not_a_command():
    assert verdict('git commit -m "Guard: deny disassembly\n\nobjdump -d libmkl_core.so is now refused"') == ALLOW
    assert verdict("cat > notes.md <<'EOF'\nstrings -a /opt/arm/lib/libarmpl_lp64.dylib found it\nEOF") == ALLOW
    assert verdict("python3 - <<'EOF'\nprint('objdump -d /opt/intel/libmkl_core.so')\nEOF") == ALLOW
    assert verdict("echo \"don't run strings on libarmpl.dylib\"") == ALLOW
    assert verdict("") == ALLOW


def test_vendor_binary_paths():
    assert is_vendor_binary("/usr/local/cuda/lib64/libcublas.so.12.8.4")
    assert is_vendor_binary("/opt/intel/oneapi/mkl/latest/lib/libmkl_core.a")
    assert is_vendor_binary("/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate")
    assert is_vendor_binary("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/cublas64_13.dll")
    assert not is_vendor_binary("/usr/local/cuda/include/cublas_v2.h")
    assert not is_vendor_binary("/System/Library/Frameworks/Accelerate.framework/Headers/cblas.h")
    assert not is_vendor_binary("/usr/lib/x86_64-linux-gnu/openblas-openmp/libopenblas.so.0")
    assert not is_vendor_binary("build/test/libscipy_shim.so")
    assert is_vendor_binary("/System/Library/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib")
    assert not is_vendor_binary("/opt/homebrew/opt/lapack/lib/liblapack.dylib")


def test_web_requests():
    assert check_web("MKL dgemm kernel disassembly blog")[0] == ASK
    assert check_web("https://example.org/leaked-cublas-source")[0] == ASK
    assert check_web("oneMKL dgemm documentation")[0] == ALLOW
    assert check_web('Intel Simplified Software License "reverse engineering" oneMKL')[0] == ALLOW
    assert check_web("reverse engineering the Eigen product kernel")[0] == ALLOW


def test_payloads():
    assert check_payload(payload("Read", file_path="/usr/local/cuda/lib64/libcublas.so.12"))[0] == DENY
    assert check_payload(payload("Read", file_path="/usr/local/cuda/include/cublas_v2.h"))[0] == ALLOW
    assert check_payload(payload("Grep", pattern="gemm", path="/opt/arm/lib/libarmpl.a"))[0] == DENY
    assert check_payload(payload("WebSearch", query="armpl decompiled sgemm"))[0] == ASK
    assert check_payload(payload("Edit", file_path="Eigen/src/Core/Dot.h"))[0] == ALLOW
    decision, reason = check_payload(payload("Bash", command="strings /opt/arm/lib/libarmpl.so"))
    assert decision == DENY and "strings" in reason and "libarmpl" in reason and ".agents/provenance.md" in reason


def run_hook(stdin_text):
    done = subprocess.run([sys.executable, SCRIPT, "--claude-hook"], input=stdin_text, capture_output=True, text=True)
    return done.returncode, done.stdout


def test_hook_protocol():
    code, out = run_hook(json.dumps(payload("Bash", command="objdump -d /opt/intel/libmkl_core.so")))
    decision = json.loads(out)["hookSpecificOutput"]
    assert code == 0 and decision["hookEventName"] == "PreToolUse" and decision["permissionDecision"] == DENY, out
    code, out = run_hook(json.dumps(payload("Bash", command="find /opt/arm -name 'libarmpl*' | xargs strings")))
    assert code == 0 and json.loads(out)["hookSpecificOutput"]["permissionDecision"] == ASK, out
    # No decision at all for ordinary work, and for input the hook cannot parse.
    assert run_hook(json.dumps(payload("Bash", command="ninja -C build dot"))) == (0, "")
    assert run_hook("not json") == (0, "")
    assert run_hook(json.dumps({"tool_name": "Bash", "tool_input": None})) == (0, "")


def test_command_mode_exit_codes():
    def code(command):
        return subprocess.run([sys.executable, SCRIPT, "--command", command], capture_output=True).returncode
    assert code("ls") == 0
    assert code("find /opt/arm -name 'libarmpl*' | xargs strings") == 1
    assert code("strings /opt/arm/lib/libarmpl.so") == 2


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS %s" % test.__name__)
    print("%d tests passed" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
