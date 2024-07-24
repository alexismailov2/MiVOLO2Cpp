cmake_minimum_required(VERSION 3.20)
project(MiVOLO2_test)

include(GNUInstallDirs)

add_executable(${PROJECT_NAME}
        main.cpp)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ../include)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        -L${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        MiVOLO2
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
