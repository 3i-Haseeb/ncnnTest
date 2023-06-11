cmake_minimum_required(VERSION 3.17)
set(CMAKE_C_COMPILER "/usr/local/opt/llvm/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/local/opt/llvm/bin/clang++")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_CXX_FLAGS
# "${CMAKE_CXX_FLAGS} -fexperimental-library -Wno-c++98-compat -Weverything
# -std=c++20 -stdlib=libc++"
# )

project(ncnnTest)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)
find_package(OpenMP REQUIRED)

set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
                opencv_videoio)

add_executable(main main.cpp utility/utility.cpp)

# target_compile_options(main PRIVATE -Wno-c++98-compat -Weverything -std=c++20
# -stdlib=libc++)
target_link_libraries(main PRIVATE ${OpenCV_LIBS} ncnn)
