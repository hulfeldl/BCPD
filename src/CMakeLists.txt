#

set(LIB_NAME ${PROJECT_NAME})

# Add executable target with source files listed in SOURCE_FILES variable
add_library(${LIB_NAME} ${SOURCE_FILES})

# Add main.cpp file of project root directory as source file
target_sources(${LIB_NAME}
    PRIVATE
    bcpd.cpp 
    PUBLIC
    bcpd.h)

target_include_directories(${LIB_NAME}
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
  )

target_link_libraries(${LIB_NAME} Eigen3::Eigen)
target_link_libraries(${LIB_NAME} CvPlot::CvPlot)