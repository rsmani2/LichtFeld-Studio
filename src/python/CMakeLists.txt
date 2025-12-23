# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

if(NOT BUILD_PYTHON_BINDINGS)
    return()
endif()

message(STATUS "Configuring Python components (runner + nanobind module)")

# Python for interpreter + embedding + nanobind
find_package(Python REQUIRED COMPONENTS Interpreter Development Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

add_library(lfs_python_utils STATIC
    runner.cpp
)

target_include_directories(lfs_python_utils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src # for includes like "python/runner.hpp"
)

target_link_libraries(lfs_python_utils
    PUBLIC
        Python::Python
        lfs_core
)

target_compile_definitions(lfs_python_utils
    PUBLIC
        LFS_BUILD_PYTHON_BINDINGS
)

set_target_properties(lfs_python_utils PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
)

# nanobind module (Python import name: lichtfeld)
nanobind_add_module(
    lfs_py
    STABLE_ABI
    NB_STATIC
    lfs/module.cpp
)

set_target_properties(lfs_py PROPERTIES
    CUDA_ARCHITECTURES ""
    CUDA_SEPARABLE_COMPILATION OFF
    LINKER_LANGUAGE CXX
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME lichtfeld
)

target_link_libraries(lfs_py PRIVATE
    lfs_core
    lfs_training
    lfs_geometry
    lfs_io
)

target_include_directories(lfs_py PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)
