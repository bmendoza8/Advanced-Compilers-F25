cmake_minimum_required(VERSION 3.13)
project(DefUseChainsPass LANGUAGES CXX)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(DefUseChains MODULE DefUseChains.cpp)

set_property(TARGET DefUseChains PROPERTY CXX_STANDARD 17)
set_target_properties(DefUseChains PROPERTIES PREFIX "")
