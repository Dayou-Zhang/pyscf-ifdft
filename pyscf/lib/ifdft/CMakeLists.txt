if(TARGET IFXC::ifxc)
  set(IFDFT_IFXC_TARGET IFXC::ifxc)
elseif(TARGET ifxc)
  set(IFDFT_IFXC_TARGET ifxc)
else()
  message(FATAL_ERROR "libifxc target not found")
endif()

add_library(ifdft_ifxc SHARED
  ifdft_ifxc.c
)

target_link_libraries(ifdft_ifxc PRIVATE ${IFDFT_IFXC_TARGET})
target_include_directories(ifdft_ifxc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ifdft_ifxc PRIVATE c_std_99)

find_package(OpenMP QUIET COMPONENTS C)
if(OpenMP_C_FOUND)
  target_link_libraries(ifdft_ifxc PRIVATE OpenMP::OpenMP_C)
endif()

set_target_properties(ifdft_ifxc PROPERTIES
  OUTPUT_NAME ifdft_ifxc
  C_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  # The empty generator expression prevents multi-config generators from
  # adding a configuration subdirectory to the in-place output location.
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/pyscf/lib$<0:>"
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/pyscf/lib$<0:>"
)

if(APPLE)
  set_target_properties(ifdft_ifxc PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path"
  )
elseif(UNIX)
  set_target_properties(ifdft_ifxc PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
endif()

add_custom_command(TARGET ifdft_ifxc POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:${IFDFT_IFXC_TARGET}>"
          "$<TARGET_FILE_DIR:ifdft_ifxc>/$<TARGET_SONAME_FILE_NAME:${IFDFT_IFXC_TARGET}>"
  VERBATIM
)

install(TARGETS ifdft_ifxc
  ARCHIVE DESTINATION pyscf/lib
  LIBRARY DESTINATION pyscf/lib
  RUNTIME DESTINATION pyscf/lib
)

install(FILES
  "$<TARGET_FILE_DIR:ifdft_ifxc>/$<TARGET_SONAME_FILE_NAME:${IFDFT_IFXC_TARGET}>"
  DESTINATION pyscf/lib
)
