#!/bin/bash
# Hexagon Toolchain Validation Script
# This script verifies that the Hexagon toolchain is properly installed and functional

set -e

echo "Hexagon Toolchain Validation"
echo "================================"

# Source environment
if [ -f /usr/local/bin/setup-hexagon-env.sh ]; then
    source /usr/local/bin/setup-hexagon-env.sh
fi

# Check toolchain binaries
echo "Checking toolchain binaries..."
REQUIRED_BINARIES=(
    "hexagon-unknown-linux-musl-clang"
    "hexagon-unknown-linux-musl-clang++"
    "qemu-hexagon"
)

for binary in "${REQUIRED_BINARIES[@]}"; do
    if [ -f "${HEXAGON_TOOLCHAIN_BIN}/${binary}" ]; then
        echo "  FOUND: ${binary}"
    else
        echo "  MISSING: ${binary}"
        exit 1
    fi
done

# Test compiler version
echo ""
echo "Checking compiler version..."
${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang --version | head -1

# Test QEMU version
echo ""
echo "Checking QEMU version..."
${HEXAGON_TOOLCHAIN_BIN}/qemu-hexagon --version | head -1

# Test sysroot
echo ""
echo "Checking sysroot..."
if [ -d "${HEXAGON_SYSROOT}" ]; then
    echo "  FOUND: Sysroot at ${HEXAGON_SYSROOT}"
    if [ -f "${HEXAGON_SYSROOT}/lib/ld-musl-hexagon.so.1" ]; then
        echo "  FOUND: Dynamic linker"
    else
        echo "  MISSING: Dynamic linker"
        exit 1
    fi
else
    echo "  MISSING: Sysroot at ${HEXAGON_SYSROOT}"
    exit 1
fi

# Test simple compilation
echo ""
echo "Testing simple compilation..."
cat > /tmp/hexagon_test.c << 'EOF'
#include <stdio.h>
int main() {
    printf("Hexagon toolchain test successful!\n");
    return 0;
}
EOF

if ${HEXAGON_TOOLCHAIN_BIN}/hexagon-unknown-linux-musl-clang /tmp/hexagon_test.c -o /tmp/hexagon_test; then
    echo "  PASS: Compilation successful"
    
    # Test execution with QEMU
    echo ""
    echo "Testing execution with QEMU..."
    if ${HEXAGON_TOOLCHAIN_BIN}/qemu-hexagon -L "${HEXAGON_SYSROOT}" /tmp/hexagon_test; then
        echo "  PASS: Execution successful"
    else
        echo "  FAIL: Execution failed"
        exit 1
    fi
else
    echo "  FAIL: Compilation failed"
    exit 1
fi

# Cleanup
rm -f /tmp/hexagon_test.c /tmp/hexagon_test

echo ""
echo "All Hexagon toolchain tests passed!"
echo "Ready for Eigen CI builds!" 