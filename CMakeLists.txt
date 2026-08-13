cmake_minimum_required(VERSION 4.0)
project(BCPD)

# Include KiSync framework
add_subdirectory(cmake/kisync)

# Add your sources
add_executable(bcpd 
    src/main.cpp
    # Add more source files here
)

# Link dependencies (add as needed)
# target_link_libraries(bcpd PRIVATE fmt::fmt spdlog::spdlog)
