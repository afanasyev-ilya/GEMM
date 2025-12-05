cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)

project(GEMM LANGUAGES CXX CUDA)

# Default to Release for single-config generators
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

find_package(CUDAToolkit REQUIRED)

# Global options target (avoid CMake CUDA20 dialect logic)
add_library(gemm_options INTERFACE)

target_compile_options(gemm_options
  INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>

    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
)

target_compile_definitions(gemm_options
  INTERFACE
    $<$<CONFIG:Release>:NDEBUG>
)

add_subdirectory(src/cuda)
