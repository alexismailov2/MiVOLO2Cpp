if (NOT EXISTS ${DEPENDENCIES_PREFIX_INSTALL}/lib/libtorch.a)
    ExternalProject_Add(libtorch_external
            URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcpu.zip
            SOURCE_DIR ${DEPENDENCIES_PREFIX_INSTALL}
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND "")
    list(APPEND dependencies_LIST libtorch_external)
endif()
