
if(NOT EIGEN_TEST_CXX_STD)
  set(EIGEN_TEST_CXX_STD CXX11)
endif()

find_package(Threads)

macro(ei_add_benchmark name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE Threads::Threads)
endmacro()

ei_add_benchmark(bench_gemm)
ei_add_benchmark(bench_sme)
