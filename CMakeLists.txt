cmake_minimum_required(VERSION 4.0)

project(BCPD)

# Add kisync-cmake.
add_subdirectory(cmake/kisync)

# Make Catch2 optional
find_package(Catch2 QUIET)

find_package(spdlog REQUIRED)
find_package(nanoflann REQUIRED)
find_package(tinyply REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CvPlot QUIET)

# Set C++ standard to C++23.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C++-specific flags.
add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
)

add_subdirectory(src)

add_subdirectory(tests)