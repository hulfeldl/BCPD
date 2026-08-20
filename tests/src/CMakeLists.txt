# CMakeLists.txt for bcpd-tests

set(EXEC_NAME "bcpd-tests")

# Add main.cpp file of project root directory as source file
set(SOURCE_FILES
        bcpd-tests.cpp)

# Add executable target with source files listed in SOURCE_FILES variable
add_executable(${EXEC_NAME} ${SOURCE_FILES})

target_include_directories(${EXEC_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(${EXEC_NAME}
    PRIVATE
        Catch2::Catch2WithMain
        bcpd-bcpd
)

add_test(NAME ${EXEC_NAME} COMMAND ${EXEC_NAME})