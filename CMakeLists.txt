cmake_minimum_required(VERSION 4.0)
project(BCPD)

add_subdirectory(cmake/kisync)

find_package(Catch2 REQUIRED)
find_package(spdlog REQUIRED)
find_package(nanoflann REQUIRED)
find_package(tinyply REQUIRED)

add_executable(bcpd 
    src/main.cpp
    src/bcpd.cpp
    src/kd-tree.cpp
)

target_link_libraries(bcpd PRIVATE 
    Catch2::Catch2WithMain 
    spdlog::spdlog
    nanoflann::nanoflann
    tinyply::tinyply
)
target_include_directories(bcpd PRIVATE src)
