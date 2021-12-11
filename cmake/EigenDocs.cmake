# This function creates a target for each file that matches `ExampleGlob`, and
# marks it as a dependency of `RootTarget`, which will be created as a custom target by this function
function(ei_make_example_targets RootTarget ExampleGlob)
  file(GLOB ExampleList "${ExampleGlob}")
  add_custom_target(${RootTarget})
  foreach (example_src ${ExampleList})
    get_filename_component(example ${example_src} NAME_WE)
    add_executable(example_${example} ${example_src})
    target_link_libraries(example_${example} PRIVATE EigenDocDeps)
    add_custom_command(
            TARGET example_${example}
            POST_BUILD
            COMMAND example_${example}
            ARGS >${CMAKE_CURRENT_BINARY_DIR}/${example}.out
    )
    add_dependencies(${RootTarget} example_${example})
  endforeach (example_src)
endfunction()

set(EIGEN_ROOT)

# This function crates a target for each file that matches `SnippetGlob`, and
# marks it as a dependency of `RootTarget`, which will be created as a custom target by this function
function(ei_make_snippet_targets RootTarget SnippetGlob)
  file(GLOB snippets_SRCS "${SnippetGlob}")

  add_custom_target(${RootTarget})

  foreach (snippet_src ${snippets_SRCS})
    get_filename_component(snippet ${snippet_src} NAME_WE)
    set(compile_snippet_target compile_${snippet})
    set(compile_snippet_src ${compile_snippet_target}.cpp)
    file(READ ${snippet_src} snippet_source_code)
    configure_file(${Eigen3_SOURCE_DIR}/doc/snippets/compile_snippet.cpp.in
            ${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src})
    add_executable(${compile_snippet_target}
            ${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src})
    target_link_libraries(${compile_snippet_target} PRIVATE EigenDocDeps)
    add_custom_command(
            TARGET ${compile_snippet_target}
            POST_BUILD
            COMMAND ${compile_snippet_target}
            ARGS >${CMAKE_CURRENT_BINARY_DIR}/${snippet}.out
    )
    add_dependencies(${RootTarget} ${compile_snippet_target})
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${compile_snippet_src}
            PROPERTIES OBJECT_DEPENDS ${snippet_src})
  endforeach (snippet_src)
endfunction()


# Add a target that gathers the common flags and dependencies for examples and snippets
add_library(EigenDocDeps INTERFACE)
target_compile_definitions(EigenDocDeps INTERFACE EIGEN_MAKING_DOCS)
target_link_libraries(EigenDocDeps INTERFACE Eigen3::Eigen)
if(EIGEN_STANDARD_LIBRARIES_TO_LINK_TO)
  target_link_libraries(EigenDocDeps INTERFACE ${EIGEN_STANDARD_LIBRARIES_TO_LINK_TO})
endif()