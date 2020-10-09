cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(LeNet5_example)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(DOWNLOAD_DATASETS)
    include(download_mnist)
endif()

add_executable(LeNet5_example training_loop.cpp convnet.cpp)
target_link_libraries(LeNet5_example "${TORCH_LIBRARIES}")
set_property(TARGET LeNet5_example PROPERTY CXX_STANDARD 14)

