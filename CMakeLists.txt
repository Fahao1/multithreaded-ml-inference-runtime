cmake_minimum_required(VERSION 3.20)
project(MultithreadedMLInferenceRuntime VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(MLRT_SANITIZER "" CACHE STRING "Optional sanitizer: address, undefined, address,undefined, or thread")
option(MLRT_PYTHON_TESTS "Register NumPy integration tests when available" ON)
option(MLRT_REQUIRE_PYTHON_TESTS "Fail configuration if Python/NumPy tests cannot be registered" OFF)
if(MLRT_REQUIRE_PYTHON_TESTS AND NOT MLRT_PYTHON_TESTS)
  message(FATAL_ERROR "Required Python tests cannot be disabled")
endif()

# Optional developer checks: only owned sources, never build trees or dependencies.
file(GLOB MLRT_CPP_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
file(GLOB MLRT_CPP_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/include/ml_runtime/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/apps/*.hpp")
find_program(MLRT_CLANG_FORMAT NAMES clang-format-17 clang-format)
if(MLRT_CLANG_FORMAT)
  add_custom_target(format-check
    COMMAND "${MLRT_CLANG_FORMAT}" --style=file --dry-run --Werror
      ${MLRT_CPP_SOURCES} ${MLRT_CPP_HEADERS}
    VERBATIM)
  add_custom_target(format
    COMMAND "${MLRT_CLANG_FORMAT}" --style=file -i ${MLRT_CPP_SOURCES} ${MLRT_CPP_HEADERS}
    VERBATIM)
endif()
find_program(MLRT_CLANG_TIDY NAMES clang-tidy-17 clang-tidy)
if(MLRT_CLANG_TIDY)
  add_custom_target(tidy-check
    COMMAND "${MLRT_CLANG_TIDY}" -p "${CMAKE_BINARY_DIR}" ${MLRT_CPP_SOURCES}
    VERBATIM)
endif()
find_package(Threads REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
  if(MLRT_SANITIZER)
    add_compile_options(-fsanitize=${MLRT_SANITIZER} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${MLRT_SANITIZER})
  endif()
elseif(MSVC)
  add_compile_options(/W4)
endif()

add_library(ml_runtime src/tensor.cpp src/model.cpp src/thread_pool.cpp src/engine.cpp)
target_include_directories(ml_runtime PUBLIC include)
target_link_libraries(ml_runtime PUBLIC Threads::Threads)
add_executable(ml_inference_cli apps/ml_inference_cli.cpp)
target_link_libraries(ml_inference_cli PRIVATE ml_runtime)

include(CTest)
if(BUILD_TESTING)
  add_executable(runtime_tests tests/runtime_tests.cpp)
  target_link_libraries(runtime_tests PRIVATE ml_runtime)
  add_test(NAME runtime_tests COMMAND runtime_tests)
  set_tests_properties(runtime_tests PROPERTIES TIMEOUT 60)
  add_test(NAME concurrency_stress COMMAND runtime_tests --stress)
  set_tests_properties(concurrency_stress PROPERTIES TIMEOUT 120)
  add_executable(metrics_tests tests/metrics_tests.cpp)
  add_test(NAME metrics_tests COMMAND metrics_tests)
  if(MLRT_PYTHON_TESTS)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_FOUND)
      execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
        RESULT_VARIABLE NUMPY_MISSING OUTPUT_QUIET ERROR_QUIET)
      if(NOT NUMPY_MISSING)
        add_test(NAME numpy_validation COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/python/validate.py --cli $<TARGET_FILE:ml_inference_cli>)
        set_tests_properties(numpy_validation PROPERTIES TIMEOUT 90)
        add_test(NAME tool_tests COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/tool_tests.py --cli $<TARGET_FILE:ml_inference_cli>)
        set_tests_properties(tool_tests PROPERTIES TIMEOUT 120)
      else()
        if(MLRT_REQUIRE_PYTHON_TESTS)
          message(FATAL_ERROR "NumPy is required for Python integration tests")
        endif()
        message(STATUS "NumPy unavailable: Python integration test not registered")
      endif()
    elseif(MLRT_REQUIRE_PYTHON_TESTS)
      message(FATAL_ERROR "Python is required for integration tests")
    endif()
  endif()
endif()
