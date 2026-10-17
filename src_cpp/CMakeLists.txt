cmake_minimum_required(VERSION 3.14)
project(dubai_segmentation_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_DIR}/include
)

link_directories(
    ${ONNXRUNTIME_DIR}/lib
)

set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/third_party/onnxruntime/lib;${ONNXRUNTIME_DIR}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

add_executable(dubai_segmentation_cpp
    src/onnx_model.cpp
    src/inference.cpp
    src/logging.cpp
    src/main.cpp
)

target_link_libraries(dubai_segmentation_cpp
    onnxruntime
    pthread
    dl
    m
)

target_compile_options(dubai_segmentation_cpp PRIVATE -O3 -Wall -Wextra)

