
if(HEXAGON_TOOLCHAIN_INCLUDED)
  return()
endif(HEXAGON_TOOLCHAIN_INCLUDED)
set(HEXAGON_TOOLCHAIN_INCLUDED true)

set(EIGEN_TEST_HVX ON)

if (NOT DSP_VERSION)
  set(DSP_VERSION v69)
endif()

set(TOOLS_VARIANT $ENV{DEFAULT_TOOLS_VARIANT})
set(PREBUILT_LIB_DIR hexagon_${TOOLS_VARIANT}_${DSP_VERSION})

# Cross Compiling for Hexagon
set(HEXAGON TRUE)
set(CMAKE_SYSTEM_NAME QURT)
set(CMAKE_SYSTEM_PROCESSOR Hexagon)
set(CMAKE_SYSTEM_VERSION "1") #${HEXAGON_PLATFORM_LEVEL})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
set(CUSTOM_RUNELF_PATH "")

# To fix backward compatibility with EAI addon.
if (NOT HEXAGON_SDK_ROOT)
    set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
endif()

if (NOT HEXAGON_TOOLS_ROOT)
    if (DEFINED ENV{HEXAGON_TOOLS_ROOT})
        set(HEXAGON_TOOLS_ROOT $ENV{HEXAGON_TOOLS_ROOT})
    endif()
    if(NOT HEXAGON_TOOLS_ROOT)
        set(HEXAGON_TOOLS_ROOT $ENV{DEFAULT_HEXAGON_TOOLS_ROOT})
    endif()
endif()

file(TO_CMAKE_PATH "${HEXAGON_TOOLS_ROOT}" HEXAGON_TOOLS_ROOT)
file(TO_CMAKE_PATH "${HEXAGON_SDK_ROOT}" HEXAGON_SDK_ROOT)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_arch.cmake)

set(HEXAGON_TOOLCHAIN ${HEXAGON_TOOLS_ROOT})
set(HEXAGON_LIB_DIR "${HEXAGON_TOOLCHAIN}/Tools/target/hexagon/lib")
set(HEXAGON_ISS_DIR ${HEXAGON_TOOLCHAIN}/Tools/lib/iss)
set(RUN_MAIN_HEXAGON "${HEXAGON_SDK_ROOT}/libs/run_main_on_hexagon/ship/${PREBUILT_LIB_DIR}/run_main_on_hexagon_sim")

set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
    HEXAGON_SDK_ROOT
    HEXAGON_TOOLS_ROOT
)

#QURT SPECIFIC LIBS and Includes
# Linker Flags
# QURT Related includes and linker flags

set(V_ARCH ${HEXAGON_ARCH})
set(_QURT_INSTALL_DIR "${HEXAGON_SDK_ROOT}/rtos/qurt/ADSP${V_ARCH}MP${V_ARCH_EXTN}")
set(_QURT_INSTALL_DIR "${HEXAGON_SDK_ROOT}/rtos/qurt/compute${V_ARCH}${V_ARCH_EXTN}")

message(DEBUG "_QURT_INSTALL_DIR:${_QURT_INSTALL_DIR}")
set(RTOS_DIR ${_QURT_INSTALL_DIR})
set(TARGET_DIR "${HEXAGON_LIB_DIR}/${V_ARCH}/G0")
include_directories(
    ${_QURT_INSTALL_DIR}/include
    ${_QURT_INSTALL_DIR}/include/qurt
    ${_QURT_INSTALL_DIR}/include/posix
    )

# Non QURT related includes and linker flags
set(TARGET_DIR_NOOS "${HEXAGON_TOOLCHAIN}/Tools/target/hexagon/lib/${HEXAGON_ARCH}")

set(EXE_LD_FLAGS
    -m${V_ARCH}
    -G0
    -fpic
    -Wl,-Bsymbolic
    -Wl,-L${TARGET_DIR_NOOS}/G0/pic
    -Wl,-L${HEXAGON_TOOLCHAIN}/Tools/target/hexagon/lib/
    -Wl,--no-threads -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign
    -shared
    "-o <TARGET> <SONAME_FLAG><TARGET_SONAME>"
    "<LINK_FLAGS>"
    -Wl,--start-group
    "<OBJECTS>"
    "<LINK_LIBRARIES>"
    -Wl,${TARGET_DIR_NOOS}/G0/pic/libc++.a
    -Wl,${TARGET_DIR_NOOS}/G0/pic/libc++abi.a
    -Wl,--end-group
    -lc
    )
    STRING(REPLACE ";" " " EXE_LD_FLAGS "${EXE_LD_FLAGS}")

set(HEXAGON_C_LINK_EXECUTABLE_LINK_OPTIONS "${EXE_LD_FLAGS}" )
message(DEBUG "Hexagon C Executable Linker Line:${HEXAGON_C_LINK_EXECUTABLE_LINK_OPTIONS}")
set(HEXAGON_CXX_LINK_EXECUTABLE_LINK_OPTIONS "${EXE_LD_FLAGS}")
message(DEBUG "Hexagon  CXX Executable Linker Line:${HEXAGON_CXX_LINK_EXECUTABLE_LINK_OPTIONS}")

# System include paths
include_directories(SYSTEM ${HEXAGON_SDK_ROOT}/incs)
include_directories(SYSTEM ${HEXAGON_SDK_ROOT}/incs/stddef)
include_directories(SYSTEM ${HEXAGON_SDK_ROOT}/ipc/fastrpc/incs)

# LLVM toolchain setup
# Compiler paths, options and architecture
set(CMAKE_C_COMPILER ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang${HEXAGON_TOOLCHAIN_SUFFIX})
set(CMAKE_CXX_COMPILER ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang++${HEXAGON_TOOLCHAIN_SUFFIX})
set(CMAKE_AR ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-ar${HEXAGON_TOOLCHAIN_SUFFIX})
set(CMAKE_ASM_COMPILER ${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-clang++${HEXAGON_TOOLCHAIN_SUFFIX})
set(HEXAGON_LINKER ${CMAKE_C_COMPILER})
set(CMAKE_PREFIX_PATH ${HEXAGON_TOOLCHAIN}/Tools/target/hexagon)
set(HEXAGON_SIM    "${HEXAGON_TOOLCHAIN}/Tools/bin/hexagon-sim${HEXAGON_TOOLCHAIN_SUFFIX}")
set(DEBUG_FLAGS "-O0 -g")
set(RELEASE_FLAGS "-O2")
set(COMMON_FLAGS "-m${HEXAGON_ARCH} -G0 -Wall -fno-zero-initialized-in-bss -fdata-sections -fpic")

set(COMMON_FLAGS "${COMMON_FLAGS} -mhvx -mhvx-length=128B")

set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-Wl,-soname,")
set(CMAKE_SHARED_LIBRARY_SONAME_CXX_FLAG "-Wl,-soname,")

set(CMAKE_CXX_FLAGS_DEBUG   "${COMMON_FLAGS} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} ${RELEASE_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG     "${COMMON_FLAGS} ${DEBUG_FLAGS}")
set(CMAKE_C_FLAGS_RELEASE   "${COMMON_FLAGS} ${RELEASE_FLAGS}")
if(ADD_SYMBOLS)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g ")
    set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE} -g ")
endif()
set(CMAKE_ASM_FLAGS_DEBUG   "${CMAKE_ASM_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_DEBUG}")
set(CMAKE_ASM_FLAGS_RELEASE "${CMAKE_ASM_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_RELEASE}")

# Linker Options
set(CMAKE_C_LINK_EXECUTABLE "${HEXAGON_LINKER} ${HEXAGON_C_LINK_EXECUTABLE_LINK_OPTIONS}")
set(CMAKE_C_CREATE_SHARED_LIBRARY "${HEXAGON_LINKER} ${HEXAGON_PIC_SHARED_LINK_OPTIONS}")
set(CMAKE_CXX_LINK_EXECUTABLE "${HEXAGON_LINKER} ${HEXAGON_CXX_LINK_EXECUTABLE_LINK_OPTIONS}")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${HEXAGON_LINKER} ${HEXAGON_PIC_SHARED_LINK_OPTIONS}")

# Run simulator
set(CUSTOM_RUNELF_PATH ${RTOS_DIR}/sdksim_bin/runelf.pbn)

set(q6ssLine1 "${HEXAGON_ISS_DIR}/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1\n")
set(q6ssLine2 "${HEXAGON_ISS_DIR}/l2vic.so 32 0xFC910000\n")
set(osamString "${RTOS_DIR}/debugger/lnx64/qurt_model.so\n")
file(WRITE ${CMAKE_BINARY_DIR}/q6ss.cfg ${q6ssLine1})
file(APPEND ${CMAKE_BINARY_DIR}/q6ss.cfg ${q6ssLine2})
file(WRITE ${CMAKE_BINARY_DIR}/osam.cfg ${osamString})

set(CMAKE_CROSSCOMPILING_EMULATOR
	${HEXAGON_SIM};-m${SIM_V_ARCH};--simulated_returnval;--usefs;${CMAKE_CURRENT_BINARY_DIR};--cosim_file;${CMAKE_BINARY_DIR}/q6ss.cfg;--l2tcm_base;0xd800;--rtos;${CMAKE_BINARY_DIR}/osam.cfg;${CUSTOM_RUNELF_PATH};--;${RUN_MAIN_HEXAGON};--)
