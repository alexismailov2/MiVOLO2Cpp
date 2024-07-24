include(ExternalProject)

if("${DEPENDENCIES_INSTALL_PREFIX}" STREQUAL "")
    set(DEPENDENCIES_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/dependencies)
endif()

set(dependencies_LIST)

include(${CMAKE_CURRENT_LIST_DIR}/Libtorch.cmake)

add_custom_target(dependencies DEPENDS ${dependencies_LIST})