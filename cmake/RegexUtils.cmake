# SPDX-License-Identifier: MPL-2.0

function(escape_string_as_regex _str_out _str_in)
  string(REGEX REPLACE "\\\\" "\\\\\\\\" FILETEST2 "${_str_in}")
  string(REGEX REPLACE "([.$+*?|-])" "\\\\\\1" FILETEST2 "${FILETEST2}")
  string(REGEX REPLACE "\\^" "\\\\^" FILETEST2 "${FILETEST2}")
  string(REGEX REPLACE "\\(" "\\\\(" FILETEST2 "${FILETEST2}")
  string(REGEX REPLACE "\\)" "\\\\)" FILETEST2 "${FILETEST2}")
  string(REGEX REPLACE "\\[" "\\\\[" FILETEST2 "${FILETEST2}")
  string(REGEX REPLACE "\\]" "\\\\]" FILETEST2 "${FILETEST2}")
  set(${_str_out} "${FILETEST2}" PARENT_SCOPE)
endfunction()

function(test_escape_string_as_regex)
  set(test1 "\\.^$-+*()[]?|")
  escape_string_as_regex(test2 "${test1}")
  set(testRef "\\\\\\.\\^\\$\\-\\+\\*\\(\\)\\[\\]\\?\\|")
  if(NOT test2 STREQUAL testRef)
	message("Error in the escape_string_for_regex function : \n   ${test1} was escaped as ${test2}, should be ${testRef}")
  endif()
endfunction()