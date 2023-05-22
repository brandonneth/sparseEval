project(SparseEval LANGUAGES CXX)

cmake_minimum_required(VERSION 3.23)

set(CMAKE_CXX_STANDARD 17)
set(BLT_CXX_STD c++17)


include(../blt/SetupBLT.cmake)

include(FindPkgConfig)
pkg_check_modules(ISL REQUIRED IMPORTED_TARGET isl)
pkg_check_modules(BARVINOK REQUIRED IMPORTED_TARGET barvinok)

find_package(RAJA REQUIRED)
add_definitions(-w)

add_subdirectory(src)
