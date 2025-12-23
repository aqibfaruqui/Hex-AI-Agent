cmake_minimum_required(VERSION 3.4...3.18)
project(mcts_cpp)

# Find Torch (needed to pass tensors between python and c++)
set(CMAKE_PREFIX_PATH "/opt/homebrew/lib/python3.11/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)

# Pybind11 (connects python and c++)
find_package(pybind11 REQUIRED)
pybind11_add_module(mcts_cpp mcts_binding.cpp)

# Link libraries
target_link_libraries(mcts_cpp PRIVATE ${TORCH_LIBRARIES})

# Mac optimisations
if(APPLE)
    target_compile_options(mcts_cpp PRIVATE -O3 -march=native)
endif()