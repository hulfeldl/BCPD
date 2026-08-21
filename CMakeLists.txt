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

# Format source code target
find_program(CLANG_FORMAT_EXE clang-format)
if(CLANG_FORMAT_EXE)
    # Collect all source files
    file(GLOB_RECURSE ALL_SOURCE_FILES
        "src/**/*.cpp"
        "src/**/*.h"
        "src/**/*.hpp"
        "src/**/*.inl"
        "tests/**/*.cpp"
        "tests/**/*.h"
        "tests/**/*.hpp"
    )

    add_custom_target(format-sources
        COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
        COMMENT "Running clang-format on all source files..."
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${ALL_SOURCE_FILES}
        COMMENT "Checking code formatting with clang-format..."
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
else()
    message(WARNING "clang-format not found. format-sources target will not be available.")
endif()