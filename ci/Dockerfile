# Eigen CI Environment with Hexagon Toolchain Support
# This Dockerfile is optimized for CI builds only
# For development environments, see eigen-tools/Dockerfile.dev

FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

# Build arguments
ARG HEXAGON_TOOLCHAIN_VERSION=20.1.4
ARG ENABLE_HEXAGON=true

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set up basic environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install essential CI packages
RUN apt-get update && \
    # Install ca-certificates first to fix SSL issues
    apt-get install -y --no-install-recommends ca-certificates && \
    # Install essential packages
    apt-get install -y --no-install-recommends \
        curl \
        wget \
        git \
        jq \
        tar \
        xz-utils \
        zstd \
        unzip && \
    # Install build tools
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        gcc \
        g++ \
        clang \
        make \
        pkg-config \
        ccache && \
    # Install runtime dependencies for QEMU
    apt-get install -y --no-install-recommends \
        libnuma1 && \
    # Install Python with pip
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-distutils && \
    # Clean up package cache
    rm -rf /var/lib/apt/lists/*

# Install latest CMake
RUN CMAKE_VERSION=$(curl -s https://api.github.com/repos/Kitware/CMake/releases/latest | jq -r '.tag_name' | sed 's/v//') && \
    wget -O cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && \
    tar -xzf cmake.tar.gz -C /opt && \
    rm cmake.tar.gz && \
    ln -s /opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/* /usr/local/bin/

# Install Hexagon toolchain (if enabled)
RUN if [ "${ENABLE_HEXAGON}" = "true" ]; then \
        echo "Installing Hexagon toolchain ${HEXAGON_TOOLCHAIN_VERSION}" && \
        TOOLCHAIN_FILE="clang+llvm-${HEXAGON_TOOLCHAIN_VERSION}-cross-hexagon-unknown-linux-musl.tar.zst" && \
        TOOLCHAIN_URL="https://artifacts.codelinaro.org/artifactory/codelinaro-toolchain-for-hexagon/${HEXAGON_TOOLCHAIN_VERSION}/${TOOLCHAIN_FILE}" && \
        echo "Downloading: ${TOOLCHAIN_URL}" && \
        mkdir -p /opt/hexagon-toolchain && \
        wget --timeout=60 --tries=3 -O /tmp/hexagon-toolchain.tar.zst "${TOOLCHAIN_URL}" && \
        # Install zstd if not present and extract
        tar --use-compress-program=zstd -xf /tmp/hexagon-toolchain.tar.zst -C /opt/hexagon-toolchain --strip-components=1 && \
        rm /tmp/hexagon-toolchain.tar.zst && \
        # Find the actual bin directory (it might be under x86_64-linux-gnu/bin)
        TOOLCHAIN_BIN=$(find /opt/hexagon-toolchain -type d -name "bin" | head -1) && \
        if [ -z "${TOOLCHAIN_BIN}" ]; then \
            echo "Error: No bin directory found in toolchain" && \
            echo "Available directories:" && \
            find /opt/hexagon-toolchain -type d && \
            exit 1 ; \
        fi && \
        echo "Found toolchain bin directory: ${TOOLCHAIN_BIN}" && \
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
        # Create symlinks in /opt/hexagon-toolchain/bin for consistency
        mkdir -p /opt/hexagon-toolchain/bin && \
        ln -sf "${TOOLCHAIN_BIN}"/* /opt/hexagon-toolchain/bin/ && \
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
ENV HEXAGON_TOOLCHAIN_BIN=/opt/hexagon-toolchain/bin
ENV HEXAGON_SYSROOT=/opt/hexagon-toolchain/x86_64-linux-gnu/target/hexagon-unknown-linux-musl
ENV PATH="${HEXAGON_TOOLCHAIN_BIN}:${PATH}"

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
    echo '# Set environment variables' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export HEXAGON_TOOLCHAIN_ROOT="/opt/hexagon-toolchain"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export HEXAGON_TOOLCHAIN_BIN="/opt/hexagon-toolchain/bin"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export HEXAGON_SYSROOT="/opt/hexagon-toolchain/x86_64-linux-gnu/target/hexagon-unknown-linux-musl"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export PATH="${HEXAGON_TOOLCHAIN_BIN}:${PATH}"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export QEMU_LD_PREFIX="${HEXAGON_SYSROOT}"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '# Export for CMake' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export EIGEN_CI_CROSS_C_COMPILER="${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'export EIGEN_CI_CROSS_CXX_COMPILER="${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang++"' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo '# Environment info' >> /usr/local/bin/setup-hexagon-env.sh && \
    echo 'echo "🔧 Hexagon Environment Setup"' >> /usr/local/bin/setup-hexagon-env.sh && \
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
LABEL architecture="x86_64" 