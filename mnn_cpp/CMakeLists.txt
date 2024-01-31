cmake_minimum_required(VERSION 3.0)
project(sam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

FIND_PACKAGE(OpenCV REQUIRED)

# include dir
# include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)
include_directories(/home/local/file/MNN_281/MNN-2.8.1/include/)
include_directories(/home/local/file/MNN_281/MNN-2.8.1/tools/cv/include/)

# libs dir
link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)
# link_directories(/home/local/file/MNN/build/libMNN.so)
# link_directories(/home/local/file/MNN/build/express/libMNN_Express.so)
# link_directories(/home/local/file/MNN/build/MNN/build/tools/cv/libMNNOpenCV.so)

message("mnn msg " ${MNN})
# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/dam.cpp)


# target
add_executable(dam ${SRCS})
target_link_libraries(dam MNN MNN_Express MNNOpenCV ${OpenCV_LIBS})


