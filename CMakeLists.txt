cmake_minimum_required(VERSION 4.0)
project(BCPD)
add_subdirectory(cmake/kisync)

# Make Catch2 optional
find_package(Catch2 QUIET)

find_package(spdlog REQUIRED)
find_package(nanoflann REQUIRED)
find_package(tinyply REQUIRED)

add_executable(bcpd
    src/main.cpp
    bcpd.cpp
)

target_compile_features(bcpd PRIVATE cxx_std_23)
target_compile_options(bcpd PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(bcpd PRIVATE
    spdlog::spdlog
    nanoflann::nanoflann
    tinyply::tinyply
)

if(Catch2_FOUND)
    target_link_libraries(bcpd PRIVATE Catch2::Catch2WithMain)
endif()

target_include_directories(bcpd PRIVATE src .)
