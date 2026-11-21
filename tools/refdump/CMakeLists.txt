# Builds the reference llama.cpp (reference/src, git-ignored) together with the refdump tool.
#   cmake -S tools/refdump -B reference/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
#   cmake --build reference/build --config Release --target refdump llama-perplexity llama-tokenize
cmake_minimum_required(VERSION 3.21)
project(refdump CXX)

set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../reference/src llama)

add_executable(refdump refdump.cpp)
target_link_libraries(refdump PRIVATE llama ggml)
target_compile_features(refdump PRIVATE cxx_std_17)
