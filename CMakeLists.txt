cmake_minimum_required(VERSION 3.9)

project(cajete-scratch)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

find_package(Cabana REQUIRED)

include_directories(SimpleCajete_lib)
add_subdirectory(SimpleCajete_lib)

add_subdirectory(Catch_tests)

add_subdirectory(examples)
