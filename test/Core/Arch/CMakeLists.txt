ei_add_test(packetmath_float "-DEIGEN_FAST_MATH=1")
ei_add_test(packetmath_integer "-DEIGEN_FAST_MATH=1")
ei_add_test(packetmath_complex "-DEIGEN_FAST_MATH=1")
ei_add_test(packetmath_special "-DEIGEN_FAST_MATH=1")
# Generic clang vector backend tests for different vector sizes.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
  typedef float v4sf __attribute__((ext_vector_type(4)));
  int main() { return __builtin_vectorelements(v4sf{}); }
" COMPILER_SUPPORTS_VECTOR_EXTENSIONS)
if(COMPILER_SUPPORTS_VECTOR_EXTENSIONS)
  ei_add_test(packetmath_generic_16 "-DEIGEN_FAST_MATH=1")
  ei_add_test(packetmath_generic_32 "-DEIGEN_FAST_MATH=1")
  ei_add_test(packetmath_generic_64 "-DEIGEN_FAST_MATH=1")
endif()
ei_add_test(packet_segment)
ei_add_test(vectorization_logic)
ei_add_test(blasutil)
ei_add_test(half_float)
ei_add_test(bfloat16_float)
