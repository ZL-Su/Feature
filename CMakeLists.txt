cmake_minimum_required(VERSION 3.17)

project(feature VERSION 1.0)
add_subdirectory(test)

file(GLOB_RECURSE LIBFEAT_SRC CONFIGURE_DEPENDS "*.h" "*.cpp" "*.c")
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
message(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native") # enable AVX support of C compiler
#set(CMAKE_SYSTEM_NAME Linux) # set target system name
#set(CMAKE_SYSTEM_PROCESSOR arm) # set target processor arch

#set_target_properties(feature PROPERTIES VERSION 1.0 SOVERSION 1)
add_library(feature_static STATIC ${LIBFEAT_SRC})
add_library(feature SHARED ${LIBFEAT_SRC})
find_package(TBB)
target_link_libraries(feature TBB::tbb)
install(TARGETS feature feature_static LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(FILES feature.h DESTINATION include/feature.h)
target_compile_features(feature_static PRIVATE cxx_std_20)
target_compile_features(feature PRIVATE cxx_std_20)