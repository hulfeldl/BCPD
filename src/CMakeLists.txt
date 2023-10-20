#

set(LIB_NAME ${PROJECT_NAME})

# Add executable target with source files listed in SOURCE_FILES variable
add_library(${LIB_NAME} STATIC)

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

  set_property(TARGET ${LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE NO)

target_link_libraries(${LIB_NAME} Eigen3::Eigen)
target_link_libraries(${LIB_NAME} tinyply::tinyply)
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  target_link_libraries(${LIB_NAME} CvPlot::CvPlot)
endif()