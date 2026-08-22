# CMakeLists.txt for bcpd library

# Set the library name based on the current source directory.
get_filename_component(DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(LIB_NAME "${DIR_NAME}-${DIR_NAME}")

# Add executable target with source files listed in SOURCE_FILES variable
add_library(${LIB_NAME} STATIC
        BCPD.cpp
        BCPD.h
        Kernel.h
        GaussianKernel.h
        KDTreeVectorOfVectorsAdaptor.h
)

set_property(TARGET ${LIB_NAME} PROPERTY POSITION_INDEPENDENT_CODE NO)

target_include_directories(${LIB_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(${LIB_NAME}
    PUBLIC
        Eigen3::Eigen
        nanoflann::nanoflann
    PRIVATE
        tinyply::tinyply
        spdlog::spdlog
)

if(CMAKE_BUILD_TYPE STREQUAL Debug AND CvPlot_FOUND)
  target_link_libraries(${LIB_NAME}
    PRIVATE
        CvPlot::CvPlot
  )
endif()