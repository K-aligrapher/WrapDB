cmake_minimum_required(VERSION 3.28)
# We don't use C++20 modules; scanning triggers an invalid Ninja rule name with CUDA on CMake 4.4.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CUDA_SCAN_FOR_MODULES OFF)
# RTX 5050 Laptop = Blackwell, compute capability 12.0 (must be set before project()).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
project(warpdb LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# ---- core library ----
add_library(warpdb_gpu STATIC src/gpu/device.cu src/gpu/hello_kernel.cu)
target_include_directories(warpdb_gpu PUBLIC src)
set_target_properties(warpdb_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# ---- CLI ----
add_executable(warpdb src/cli/main.cpp)
target_link_libraries(warpdb PRIVATE warpdb_gpu)

# ---- tests ----
include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(warpdb_tests tests/test_device.cpp)
target_link_libraries(warpdb_tests PRIVATE warpdb_gpu GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(warpdb_tests)
