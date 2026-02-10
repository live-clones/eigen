#!/bin/bash
set -e

# Detect number of cores
THREADS=$(sysctl -n hw.logicalcpu)
export OMP_NUM_THREADS=$THREADS

echo "Detected $THREADS threads. Enabling OpenMP..."

# OpenMP paths for the standalone libomp package
OMP_FLAGS="-I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -fopenmp -Wno-macro-redefined"

echo "Compiling NEON benchmark (Standard M4)..."
/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang++ -O3 -std=c++17 -I. -Ibench -mcpu=apple-m4 $OMP_FLAGS bench/bench_sme.cpp -o bench/bench_sme_neon

echo "Compiling SME benchmark (M4 SME)..."
/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang++ -O3 -std=c++17 -I. -Ibench -mcpu=apple-m4+sme+nosve $OMP_FLAGS \
    -DEIGEN_VECTORIZE_SME -DEIGEN_ARM64_USE_SME -DEIGEN_SME_USE_NEON_PACKETS \
    bench/bench_sme.cpp -o bench/bench_sme_sme

echo "Running Comparison (Size: 4096)..."
echo "-----------------------------------"
echo "NEON Implementation ($THREADS threads):"
./bench/bench_sme_neon 4096 4
echo "-----------------------------------"
echo "SME Implementation ($THREADS threads):"
./bench/bench_sme_sme 4096 4
