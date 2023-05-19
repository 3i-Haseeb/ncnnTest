cmake_minimum_required(VERSION 3.17)
project(ncnnTest)

set(CMAKE_CXX_STANDARD 17)

set(OpenMP_DIR /usr/local/opt/libomp)
set(OpenCV_DIR /usr/local/include/opencv4)
set(NCNN_DIR
    /Users/3i-a1-2021-15/Developer/projects/pivo-tracking/ncnn/build/install/)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED HINTS ${NCNN_DIR})

# Enable OpenMP support in your CMake project by setting the appropriate
# compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include the headers
include_directories(classifier PRIVATE ${OpenCV_INCLUDE_DIRS}
                    ${NCNN_INCLUDE_DIRS} ${OpenMP_DIR}/include)
link_directories(${OpenMP_DIR}/lib)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE ${OpenCV_LIBS} ncnn "-lomp")
