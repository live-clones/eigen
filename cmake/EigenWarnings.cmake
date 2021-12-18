# this file creates an interface líbrary EigenWarnings that can be used to enable
# the warnings we want in tests and examples
add_library(EigenWarnings INTERFACE)

function(ei_add_warning_flag FLAG)
    # generate a valid CMake variable name from FLAG to store the result of the test
    string(REGEX REPLACE "-" "" SFLAG1 ${FLAG})
    string(REGEX REPLACE "\\+" "p" SFLAG ${SFLAG1})
    check_cxx_compiler_flag(${FLAG} COMPILER_SUPPORT_${SFLAG})
    if(COMPILER_SUPPORT_${SFLAG})
        target_compile_options(EigenWarnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
    endif()
endfunction()

if(NOT MSVC)
    # We assume that other compilers are partly compatible with GNUCC

    # clang outputs some warnings for unknown flags that are not caught by check_cxx_compiler_flag
    # adding -Werror turns such warnings into errors
    check_cxx_compiler_flag("-Werror" COMPILER_SUPPORT_WERROR)
    if(COMPILER_SUPPORT_WERROR)
        # TODO for CUDA -Werror seems to require an argument
        target_compile_options(EigenWarnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
    endif()

    # TODO the old code added all of them, and then manually mucked about in the CMAKE_CXX_FLAGS for cuda
    # TODO Right now, all of these warnings are not set for CUDA. The old comment:
    # Make sure to compile without the -pedantic, -Wundef, -Wnon-virtual-dtor
    # and -fno-check-new flags since they trigger thousands of compilation warnings
    # in the CUDA runtime
    ei_add_warning_flag("-pedantic")        # not in CUDA
    ei_add_warning_flag("-Wall")
    ei_add_warning_flag("-Wextra")
    #ei_add_warning_flag("-Weverything")              # clang

    ei_add_warning_flag("-Wundef")          # not in CUDA
    ei_add_warning_flag("-Wcast-align")
    ei_add_warning_flag("-Wchar-subscripts")
    ei_add_warning_flag("-Wnon-virtual-dtor")  # not in CUDA
    ei_add_warning_flag("-Wunused-local-typedefs")
    ei_add_warning_flag("-Wpointer-arith")
    ei_add_warning_flag("-Wwrite-strings")
    ei_add_warning_flag("-Wformat-security")
    ei_add_warning_flag("-Wshorten-64-to-32")
    ei_add_warning_flag("-Wlogical-op")
    ei_add_warning_flag("-Wenum-conversion")
    ei_add_warning_flag("-Wc++11-extensions")
    ei_add_warning_flag("-Wdouble-promotion")
    #  ei_add_warning_flag("-Wconversion")

    ei_add_warning_flag("-Wshadow")

    ei_add_warning_flag("-Wno-psabi")
    ei_add_warning_flag("-Wno-variadic-macros")
    ei_add_warning_flag("-Wno-long-long")

    ei_add_warning_flag("-fno-check-new")            # not in CUDA
    ei_add_warning_flag("-fno-common")
    ei_add_warning_flag("-fstrict-aliasing")
    ei_add_warning_flag("-wd981")                    # disable ICC's "operands are evaluated in unspecified order" remark
    ei_add_warning_flag("-wd2304")                   # disable ICC's "warning #2304: non-explicit constructor with single argument may cause implicit type conversion" produced by -Wnon-virtual-dtor

    if(ANDROID_NDK)
        ei_add_warning_flag("-pie")
        ei_add_warning_flag("-fPIE")
    endif()
else()
    # C4127 - conditional expression is constant
    # C4714 - marked as __forceinline not inlined (I failed to deactivate it selectively)
    #         We can disable this warning in the unit tests since it is clear that it occurs
    #         because we are oftentimes returning objects that have a destructor or may
    #         throw exceptions - in particular in the unit tests we are throwing extra many
    #         exceptions to cover indexing errors.
    # C4505 - unreferenced local function has been removed (impossible to deactivate selectively)
    target_compile_options(EigenWarnings INTERFACE /EHsc /wd4127 /wd4505 /wd4714)
endif()