cmake_minimum_required(VERSION 3.20)
project(MiVOLO2)

set(CMAKE_CXX_STANDARD 17)

include(dependencies/dependencies.cmake)

message(STATUS "TORCH_PATH=${CMAKE_BINARY_DIR}/${DEPENDENCIES_PREFIX_INSTALL}/share/cmake/Torch")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")

find_package(Torch QUIET PATHS ${CMAKE_BINARY_DIR}/dependencies/share/cmake/Torch)
message(STATUS "TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")

add_library(${PROJECT_NAME}
        src/Yolov8.cpp
        src/AgeAndGender.cpp)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        include)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES})

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/MiVOLO2 DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(tests)