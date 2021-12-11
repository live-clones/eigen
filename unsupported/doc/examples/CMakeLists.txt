include_directories(../../../unsupported ../../../unsupported/test)
ei_make_example_targets(unsupported_examples "*.cpp")

if(EIGEN_TEST_SYCL)
  add_subdirectory(SYCL)
endif(EIGEN_TEST_SYCL)
