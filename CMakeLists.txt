set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(LIBFEAT_SRC src/feature.cpp)
cmake_minimum_required(VERSION 3.17)

project(feature LANGUAGE CXX)

set_target_properties(feature PROPERTIES VERSION 1.0 SOVERSION 1)

add_library(feature_static STATIC ${LIBFEAT_SRC})
add_library(feature SHARED ${LIBFEAT_SRC})
install(TARGET feature feature_static
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib)
install(FILES feature.h DESTINATION include/feature.h)