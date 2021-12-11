set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

ei_make_snippet_targets(unsupported_snippets "snippets/*.cpp")
ei_make_example_targets(unsupported_examples "examples/*.cpp")

if(EIGEN_TEST_SYCL)
  add_subdirectory(examples/SYCL)
endif(EIGEN_TEST_SYCL)
