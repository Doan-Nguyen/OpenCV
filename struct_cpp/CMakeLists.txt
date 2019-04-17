cmake_minimum_required(VERSION 3.9)
project(pg_c)

set(CMAKE_CXX_STANDARD 11)

find_package(dlib REQUIRED)

add_executable(
    dlib
    dlib/main.cpp
)

target_link_ libraries(
    dlib
    dlib::dlib
)

