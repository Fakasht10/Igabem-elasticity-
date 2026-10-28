cmake_minimum_required(VERSION 3.16)
project(igabem_elasticity CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

add_executable(patch_test app/patch_test.cpp)
target_include_directories(patch_test PRIVATE src)
target_link_libraries(patch_test PRIVATE Eigen3::Eigen)
target_compile_options(patch_test PRIVATE -O3 -march=native -Wall -Wextra)
if(OpenMP_CXX_FOUND)
  target_link_libraries(patch_test PRIVATE OpenMP::OpenMP_CXX)
endif()
