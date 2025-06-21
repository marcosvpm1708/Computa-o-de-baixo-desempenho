cmake_minimum_required(VERSION 3.12)
project(cuda_monte_carlo LANGUAGES CXX CUDA)

# Cria somente a biblioteca compartilhada libmonte_carlo.so
add_library(monte_carlo SHARED
    mc_wrapper.cu
    mc_kernel.cu
)

target_include_directories(monte_carlo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(monte_carlo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -std=c++17>
)

set_target_properties(monte_carlo PROPERTIES
    CUDA_ARCHITECTURES "61"
    OUTPUT_NAME "monte_carlo"
)
