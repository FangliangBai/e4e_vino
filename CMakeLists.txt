cmake_minimum_required(VERSION 3.21)
project(e4e_vino)
set(CMAKE_CXX_STANDARD 14)

# OpenVINO
find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)

# Dlib
add_subdirectory(./dlib-19.23)

# ONNXRuntime
set(onnxruntime_INCLUDE_DIRS ./onnxruntime-linux-x64-gpu-1.10.0/include)
include_directories(${onnxruntime_INCLUDE_DIRS})

add_executable(e4e_vino main.cc FaceAligner.h FaceAligner.cpp)

target_include_directories(e4e_vino PRIVATE ${PROJECT_SOURCE_DIR}/)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE ${InferenceEngine_LIBRARIES}
    ${OpenCV_LIBS}
    ${NGRAPH_LIBRARIES}
    dlib::dlib
    /media/kent/DISK2/CLionProjects/openvino/onnxruntime-linux-x64-gpu-1.10.0/lib/libonnxruntime.so
    /media/kent/DISK2/CLionProjects/openvino/onnxruntime-linux-x64-gpu-1.10.0/lib/libonnxruntime_providers_cuda.so
)