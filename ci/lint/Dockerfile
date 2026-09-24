# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# CI image for the checkformat lint jobs and select:tests.
#
# Kept outside ci/docker/ on purpose: everything there is hashed into the test
# pass-cache fingerprint and forces the full affected-test suite, and nothing
# in this image can change a test's outcome.
#
# clang-format is built from the LLVM release rather than taken from a package:
# Alpine dropped clang17 after 3.22, and the PyPI musllinux wheel's
# patchelf-rewritten binary does not load under musl.  The static build is
# ~3 MB against ~300 MB for Alpine's clang17-extra-tools.
#
# Rebuild and push when this file changes:
#   docker buildx build --platform linux/amd64 \
#     -t registry.gitlab.com/libeigen/eigen/alpine-lint:latest \
#     --push ci/lint/

FROM alpine:3.24 AS clang-format

ARG LLVM_VERSION=17.0.6
# Matches the sha512 in Alpine's llvm17 APKBUILD.
ARG LLVM_SHA256=58a8818c60e6627064f312dbf46c02d9949956558340938b71cf731ad8bc0813

RUN apk add --no-cache build-base cmake samurai python3 curl

WORKDIR /src
RUN curl -fsSLo llvm.tar.xz \
      https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz \
    && echo "${LLVM_SHA256}  llvm.tar.xz" | sha256sum -c - \
    && tar -xJf llvm.tar.xz --strip-components=1 \
    && rm llvm.tar.xz

# LLVM 17 predates GCC 15, whose libstdc++ no longer includes <cstdint>
# transitively.
RUN cmake -S llvm -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=MinSizeRel \
      -DCMAKE_CXX_FLAGS="-include cstdint" \
      -DCMAKE_EXE_LINKER_FLAGS="-static -s" \
      -DLLVM_ENABLE_PROJECTS=clang \
      -DLLVM_TARGETS_TO_BUILD=Native \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF \
      -DCLANG_INCLUDE_TESTS=OFF -DCLANG_INCLUDE_DOCS=OFF \
      -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF \
      -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
    && samu -C build clang-format \
    && install -D -m 755 build/bin/clang-format /out/clang-format \
    && install -D -m 755 clang/tools/clang-format/git-clang-format /out/git-clang-format

FROM alpine:3.24

# After populating the venv, drop what no job reaches (~9 MB compressed):
#  - pip, ensurepip, and the stdlib's GUI, REPL-help, test and build parts;
#  - git's HTTP transport and libcurl's dependency chain: the runner's helper
#    image does the clone, and the jobs only read the local repository;
#  - OpenSSL, and with it apk: hashlib falls back to the builtin _sha2/_md5;
#  - the C decimal accelerator (and libstdc++ it alone needs), and sqlite3.
RUN apk add --no-cache python3 git \
    && python3 -m venv /opt/lint \
    && /opt/lint/bin/pip install --no-cache-dir \
      codespell==2.2.6 \
      'reuse[charset-normalizer]==6.2.0' \
    && /opt/lint/bin/pip uninstall -q -y pip \
    && py=$(echo /usr/lib/python3.*) && dyn=$py/lib-dynload \
    && rm -rf $py/ensurepip $py/idlelib $py/tkinter $py/turtledemo $py/turtle.py \
      $py/pydoc_data $py/test $py/config-3.* $py/__phello__ $py/sqlite3 \
      $dyn/_test*.so $dyn/_ctypes_test*.so $dyn/xxlimited*.so $dyn/_xxtestfuzz*.so \
      $dyn/_decimal*.so $dyn/_hashlib*.so $dyn/_ssl*.so $dyn/_sqlite3*.so \
      /usr/libexec/git-core/git-remote-http* /usr/libexec/git-core/git-remote-ftp* \
      /usr/libexec/git-core/git-http-* /usr/libexec/git-core/git-imap-send \
      /usr/lib/libcurl.so* /usr/lib/libbrotli*.so* /usr/lib/libnghttp2.so* \
      /usr/lib/libcares.so* /usr/lib/libidn2.so* /usr/lib/libunistring.so* \
      /usr/lib/libpsl.so* /usr/lib/libmpdec.so* /usr/lib/libstdc++.so* \
      /usr/lib/libgcc_s.so* /usr/lib/libsqlite3.so* \
      /usr/lib/libcrypto.so* /usr/lib/libssl.so* /lib/libcrypto.so* /lib/libssl.so* \
      /usr/bin/ssl_client /sbin/apk /usr/lib/libapk.so* /lib/libapk.so* \
      /lib/apk /etc/apk /usr/share/apk /etc/ssl* /usr/share/ca-certificates

COPY --from=clang-format /out/ /usr/local/bin/

ENV PATH=/opt/lint/bin:$PATH
