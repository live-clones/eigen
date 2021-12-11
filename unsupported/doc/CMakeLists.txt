set_directory_properties(PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_subdirectory(examples)
ei_make_snippet_targets(unsupported_snippets "snippets/*.cpp")
