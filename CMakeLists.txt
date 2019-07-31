cmake_minimum_required(VERSION 3.4.3)

project(SYCL_Examples)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

option(COMPUTECPP_SDK_USE_OPENMP "Enable OpenMP support in samples" OFF)
option(COMPUTECPP_SDK_BUILD_TESTS "Build the tests for the header utilities in include/" OFF)

set(CMAKE_CXX_STANDARD 14)
enable_testing()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

function(add_sample)
    set(options NO_TEST)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(SDK_ADD_SAMPLE
      "${options}"
      "${one_value_args}"
      "${multi_value_args}"
      ${ARGN}
    )
    add_executable(${SDK_ADD_SAMPLE_TARGET} ${SDK_ADD_SAMPLE_SOURCES})
    add_sycl_to_target(TARGET ${SDK_ADD_SAMPLE_TARGET}
      SOURCES ${SDK_ADD_SAMPLE_SOURCES})
    if(NOT SDK_ADD_SAMPLE_NO_TEST)
      add_test(NAME ${SDK_ADD_SAMPLE_TARGET} COMMAND ${SDK_ADD_SAMPLE_TARGET})
    endif()
    install(TARGETS ${SDK_ADD_SAMPLE_TARGET} RUNTIME DESTINATION bin)
endfunction(add_sample)

add_sample(TARGET hello_world SOURCES src/hello_world.cpp)
add_sample(TARGET error_handling SOURCES src/error_handling.cpp)
add_sample(TARGET memory1 SOURCES src/memory1.cpp)