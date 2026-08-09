cmake_minimum_required(VERSION 4.0)

if(EXISTS "${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
    include("${CMAKE_BINARY_DIR}/conan_toolchain.cmake")
endif()

#
#
#

cmake_minimum_required(VERSION 3.24)  # CMake version check
project(bcpd)               # Create project "simple_example"
set(CMAKE_CXX_STANDARD 20)            # Enable c++20 standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages.
find_package(Eigen3 REQUIRED)
find_package(Catch2 REQUIRED)
find_package(tinyply REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL Debug)
    find_package(CvPlot REQUIRED)
endif()


include(GNUInstallDirs)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
add_subdirectory(external/cmake)
target_link_libraries(BCPD PRIVATE KiSyncLib fmt::fmt spdlog::spdlog)
