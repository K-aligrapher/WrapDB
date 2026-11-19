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
                    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
                    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177>)  # 177: unused gtest members

# ---- core library ----
find_package(CUDAToolkit REQUIRED)
add_library(warpdb_gpu STATIC src/gpu/device.cu src/gpu/hello_kernel.cu src/gpu/gpu_backend.cu
  src/gpu/jit/jit.cu src/gpu/jit/codegen.cpp)
# JIT-generated kernels #include the device-only headers from src/ through NVRTC at run time
target_compile_definitions(warpdb_gpu PRIVATE WARPDB_SRC_DIR="${CMAKE_SOURCE_DIR}/src")
target_link_libraries(warpdb_gpu PUBLIC CUDA::nvrtc CUDA::cuda_driver)
target_include_directories(warpdb_gpu PUBLIC src)
set_target_properties(warpdb_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_link_libraries(warpdb_gpu PUBLIC warpdb_core)

# ---- CPU engine core (storage, SQL front end, executor) ----
add_library(warpdb_core STATIC
  src/storage/csv.cpp
  src/sql/parser.cpp
  src/exec/eval.cpp
  src/exec/engine.cpp
  src/exec/placement.cpp
  src/exec/calibrate.cpp
  src/optimizer/cost_model.cpp)
target_include_directories(warpdb_core PUBLIC src)
if(MSVC)
  target_compile_options(warpdb_core PRIVATE /W3 /O2 /EHsc)
endif()

# ---- CLI ----
add_executable(warpdb src/cli/main.cpp)
target_link_libraries(warpdb PRIVATE warpdb_core warpdb_gpu)

# ---- Python module (pybind11): `import warpdb` from python/ ----
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
if(Python3_FOUND)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
                  OUTPUT_VARIABLE WARPDB_PYBIND11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _pyb_rc)
  if(_pyb_rc EQUAL 0)
    find_package(pybind11 CONFIG QUIET PATHS "${WARPDB_PYBIND11_DIR}" NO_DEFAULT_PATH)
  endif()
endif()
if(pybind11_FOUND)
  pybind11_add_module(_warpdb MODULE src/python/bindings.cpp)
  target_link_libraries(_warpdb PRIVATE warpdb_core warpdb_gpu)
  set_target_properties(_warpdb PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/warpdb"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/warpdb"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/warpdb"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/warpdb")
  message(STATUS "Python module _warpdb enabled (pybind11 ${pybind11_VERSION})")
else()
  message(STATUS "pybind11 not found: Python module disabled (pip install pybind11)")
endif()

# ---- GPU micro-benchmarks (WarpDB primitives vs CUB; CUB is a reference only) ----
add_executable(warpdb_bench bench/bench_gpu.cu)
target_include_directories(warpdb_bench PRIVATE src)
if(MSVC)  # CUB (CCCL 3) requires the conforming preprocessor
  target_compile_options(warpdb_bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

# ---- tests ----
include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(warpdb_tests tests/test_device.cpp tests/test_sql.cpp tests/test_gpu_prims.cu tests/test_gpu_engine.cu tests/test_optimizer.cpp tests/test_jit.cu)
target_link_libraries(warpdb_tests PRIVATE warpdb_core warpdb_gpu GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(warpdb_tests)
