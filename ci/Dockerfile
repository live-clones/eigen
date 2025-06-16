# Eigen CI Environment with Hexagon Toolchain Support
# This Dockerfile is optimized for CI builds only
# For development environments, see eigen-tools/Dockerfile.dev

FROM ubuntu:20.04

# Build arguments
ARG HEXAGON_TOOLCHAIN_VERSION=19.1.5
ARG ENABLE_HEXAGON=true

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set up basic environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Architecture detection and setup for CI builds  
RUN TARGET_ARCH=$(uname -m) && \
    echo "Detected architecture: ${TARGET_ARCH}" && \
    case "${TARGET_ARCH}" in \
        "x86_64") \
            echo "x86_64" > /tmp/target_arch && \
            echo "linux-x86_64" > /tmp/cmake_arch && \
            echo "linux-amd64" > /tmp/bazel_arch ;; \
        "aarch64") \
            echo "aarch64" > /tmp/target_arch && \
            echo "linux-aarch64" > /tmp/cmake_arch && \
            echo "linux-arm64" > /tmp/bazel_arch ;; \
        *) \
            echo "Unsupported architecture: ${TARGET_ARCH}" && exit 1 ;; \
    esac

# Install essential CI packages only (skip ca-certificates update for x86_64 emulation)
RUN apt-get update && \
    # Install packages without ca-certificates first
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        wget \
        git \
        jq \
        tar \
        xz-utils \
        zstd \
        unzip && \
    # Install build tools
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        gcc \
        g++ \
        clang \
        make \
        pkg-config \
        ccache && \
    # Install Python (minimal)
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 \
        python3-distutils && \
    # Clean up package cache
    rm -rf /var/lib/apt/lists/* && \
    # Install pip separately using get-pip for Python 3.8 (skip SSL verification for x86_64 emulation)
    wget --no-check-certificate -O /tmp/get-pip.py https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python3 /tmp/get-pip.py --no-cache-dir && \
    rm /tmp/get-pip.py

# Install latest CMake (architecture-aware)
RUN TARGET_ARCH=$(cat /tmp/target_arch) && \
    CMAKE_ARCH=$(cat /tmp/cmake_arch) && \
    CMAKE_VERSION=$(curl -s -k https://api.github.com/repos/Kitware/CMake/releases/latest | jq -r '.tag_name' | sed 's/v//') && \
    wget --no-check-certificate -O cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-${CMAKE_ARCH}.tar.gz && \
    tar -xzf cmake.tar.gz -C /opt && \
    rm cmake.tar.gz && \
    ln -s /opt/cmake-${CMAKE_VERSION}-${CMAKE_ARCH}/bin/* /usr/local/bin/

# Install Hexagon toolchain (if enabled)
RUN if [ "${ENABLE_HEXAGON}" = "true" ]; then \
        TARGET_ARCH=$(cat /tmp/target_arch) && \
        echo "Installing Hexagon toolchain ${HEXAGON_TOOLCHAIN_VERSION} for ${TARGET_ARCH}" && \
        # Determine correct filename based on architecture and version
        if [ "${TARGET_ARCH}" = "aarch64" ]; then \
            TOOLCHAIN_FILE="clang+llvm-${HEXAGON_TOOLCHAIN_VERSION}-cross-hexagon-unknown-linux-musl_aarch64-linux-gnu.tar.zst" ; \
            TOOLCHAIN_BIN="/opt/hexagon-toolchain/aarch64-linux-musl/bin" ; \
        else \
            # For x86_64, use the Ubuntu 20.04 version (correct path is just bin/)
            TOOLCHAIN_FILE="clang+llvm-${HEXAGON_TOOLCHAIN_VERSION}-cross-hexagon-unknown-linux-musl.tar.zst" ; \
            TOOLCHAIN_BIN="/opt/hexagon-toolchain/bin" ; \
        fi && \
        TOOLCHAIN_URL="https://github.com/quic/toolchain_for_hexagon/releases/download/v${HEXAGON_TOOLCHAIN_VERSION}/${TOOLCHAIN_FILE}" && \
        echo "Downloading: ${TOOLCHAIN_URL}" && \
        mkdir -p /opt/hexagon-toolchain && \
        wget --no-check-certificate --timeout=60 --tries=3 -O /tmp/hexagon-toolchain.tar.zst "${TOOLCHAIN_URL}" && \
        # Install zstd if not present and extract
        tar --use-compress-program=zstd -xf /tmp/hexagon-toolchain.tar.zst -C /opt/hexagon-toolchain --strip-components=1 && \
        rm /tmp/hexagon-toolchain.tar.zst && \
        # Verify toolchain structure  
        if [ ! -d "${TOOLCHAIN_BIN}" ]; then \
            echo "Error: Expected toolchain binary directory not found: ${TOOLCHAIN_BIN}" && \
            echo "Available directories:" && \
            find /opt/hexagon-toolchain -type d -name "bin" && \
            exit 1 ; \
        fi && \
        # Verify key binaries exist
        for binary in hexagon-unknown-linux-musl-clang hexagon-unknown-linux-musl-clang++; do \
            if [ ! -f "${TOOLCHAIN_BIN}/${binary}" ]; then \
                echo "Error: Required Hexagon binary not found: ${TOOLCHAIN_BIN}/${binary}" && \
                echo "Available binaries:" && \
                ls -la "${TOOLCHAIN_BIN}/" && \
                exit 1 ; \
            fi ; \
        done && \
        # Check for qemu-hexagon (may not be present in all versions)
        if [ -f "${TOOLCHAIN_BIN}/qemu-hexagon" ]; then \
            echo "QEMU Hexagon found" ; \
        else \
            echo "QEMU Hexagon not found (may not be available in this version)" ; \
        fi && \
        echo "Hexagon toolchain installed successfully" ; \
    else \
        echo "Skipping Hexagon toolchain installation" ; \
    fi

# Install minimal Python packages for CI
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        pytest \
        pytest-cov \
        numpy

# Set up environment variables
ENV HEXAGON_TOOLCHAIN_VERSION=${HEXAGON_TOOLCHAIN_VERSION}
ENV HEXAGON_TOOLCHAIN_ROOT=/opt/hexagon-toolchain

# Architecture-aware environment setup
RUN TARGET_ARCH=$(cat /tmp/target_arch) && \
    if [ "${TARGET_ARCH}" = "aarch64" ]; then \
        echo 'export HEXAGON_TOOLCHAIN_BIN="/opt/hexagon-toolchain/aarch64-linux-musl/bin"' >> /etc/environment && \
        echo 'export HEXAGON_SYSROOT="/opt/hexagon-toolchain/aarch64-linux-musl/target/hexagon-unknown-linux-musl"' >> /etc/environment ; \
    else \
        echo 'export HEXAGON_TOOLCHAIN_BIN="/opt/hexagon-toolchain/bin"' >> /etc/environment && \
        echo 'export HEXAGON_SYSROOT="/opt/hexagon-toolchain/target/hexagon-unknown-linux-musl"' >> /etc/environment ; \
    fi && \
    echo 'export PATH="${HEXAGON_TOOLCHAIN_BIN}:${PATH}"' >> /etc/environment

# Set up ccache for CI builds
RUN mkdir -p /root/.ccache && \
    echo 'max_size = 2.0G' > /root/.ccache/ccache.conf && \
    echo 'compression = true' >> /root/.ccache/ccache.conf && \
    echo 'sloppiness = file_macro,locale,time_macros' >> /root/.ccache/ccache.conf

# Create setup script for runtime environment
RUN echo '#!/bin/bash' > /usr/local/bin/setup-hexagon-env.sh && \
    echo '# Runtime setup script for Hexagon environment' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'set -e' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '# Detect architecture' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'TARGET_ARCH=$(uname -m)' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'if [ "${TARGET_ARCH}" = "aarch64" ]; then' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '    export HEXAGON_TOOLCHAIN_BIN="/opt/hexagon-toolchain/aarch64-linux-musl/bin"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '    export HEXAGON_SYSROOT="/opt/hexagon-toolchain/aarch64-linux-musl/target/hexagon-unknown-linux-musl"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'else' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '    export HEXAGON_TOOLCHAIN_BIN="/opt/hexagon-toolchain/bin"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '    export HEXAGON_SYSROOT="/opt/hexagon-toolchain/target/hexagon-unknown-linux-musl"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'fi' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '# Set environment variables' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export HEXAGON_TOOLCHAIN_ROOT="/opt/hexagon-toolchain"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export PATH="${HEXAGON_TOOLCHAIN_BIN}:${PATH}"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export QEMU_LD_PREFIX="${HEXAGON_SYSROOT}"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '# Export for CMake' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export EIGEN_CI_CROSS_C_COMPILER="${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export EIGEN_CI_CROSS_CXX_COMPILER="${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang++"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '# Architecture info' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'echo "🔧 Hexagon Environment Setup"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'echo "Architecture: ${TARGET_ARCH}"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'echo "Toolchain: ${HEXAGON_TOOLCHAIN_ROOT}"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'echo "Compilers: ${EIGEN_CI_CROSS_C_COMPILER}"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'echo "Sysroot: ${HEXAGON_SYSROOT}"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '# Verify setup' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'if [ -f "${EIGEN_CI_CROSS_C_COMPILER}" ]; then' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '    echo "✅ Hexagon toolchain ready"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'else' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '    echo "❌ Hexagon toolchain not found"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '    exit 1' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'fi' >> /usr/local/bin/setup-hexagon-env.sh

RUN chmod +x /usr/local/bin/setup-hexagon-env.sh

# Set working directory for CI
WORKDIR /workspace

# Default command for CI
CMD ["/bin/bash"]

# Metadata labels for CI
LABEL maintainer="eigen-ci-team"
LABEL description="Eigen CI Environment with Hexagon Toolchain Support"
LABEL version="1.0"
LABEL hexagon.toolchain.version="${HEXAGON_TOOLCHAIN_VERSION}"
LABEL purpose="ci"
LABEL architecture="multi-arch" 