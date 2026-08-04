cmake_minimum_required(VERSION 3.16)

project(pyscf_ifdft LANGUAGES C)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BUILD_TESTING OFF CACHE BOOL "Disable libifxc tests when vendored" FORCE)

if(DEFINED ENV{IFDFT_LIBIFXC_SOURCE_DIR})
  if(NOT EXISTS "$ENV{IFDFT_LIBIFXC_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "IFDFT_LIBIFXC_SOURCE_DIR does not point to a libifxc source tree: "
      "$ENV{IFDFT_LIBIFXC_SOURCE_DIR}")
  endif()
  add_subdirectory("$ENV{IFDFT_LIBIFXC_SOURCE_DIR}" libifxc-build EXCLUDE_FROM_ALL)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pyscf/lib/deps/libifxc/CMakeLists.txt")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/pyscf/lib/deps/libifxc" libifxc-build EXCLUDE_FROM_ALL)
else()
  find_package(IFXC CONFIG QUIET)
  if(NOT TARGET IFXC::ifxc AND NOT TARGET ifxc)
    include(FetchContent)
    find_package(Git REQUIRED)
    FetchContent_Declare(
      libifxc
      GIT_REPOSITORY https://github.com/Dayou-Zhang/libifxc.git
      GIT_TAG main
      GIT_SHALLOW TRUE
      GIT_PROGRESS TRUE
    )
    FetchContent_GetProperties(libifxc)
    if(NOT libifxc_POPULATED)
      FetchContent_Populate(libifxc)
      add_subdirectory(
        "${libifxc_SOURCE_DIR}"
        "${libifxc_BINARY_DIR}"
        EXCLUDE_FROM_ALL
      )
    endif()
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
      WORKING_DIRECTORY "${libifxc_SOURCE_DIR}"
      OUTPUT_VARIABLE IFDFT_LIBIFXC_GIT_REVISION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE IFDFT_LIBIFXC_GIT_RESULT
    )
    if(NOT IFDFT_LIBIFXC_GIT_RESULT EQUAL 0)
      message(FATAL_ERROR "Could not identify the fetched libifxc revision")
    endif()
    message(STATUS
      "Fetched libifxc main at ${IFDFT_LIBIFXC_GIT_REVISION}")
  endif()
endif()

add_subdirectory(pyscf/lib)
