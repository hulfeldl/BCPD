#
#
#

cmake_minimum_required(VERSION 3.24)  # CMake version check
project(bcpd)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 20)            # Enable c++20 standard

# Find packages.
find_package(Eigen3 REQUIRED)
find_package(Catch2 REQUIRED)
find_package(CvPlot REQUIRED)


include(GNUInstallDirs)


enable_testing()

add_subdirectory(src)
add_subdirectory(tests)