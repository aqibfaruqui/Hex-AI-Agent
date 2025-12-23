from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='mcts_cpp',
    ext_modules=[
        CppExtension(
            name='agents.Group41.mcts_cpp',
            sources=['agents/Group41/cpp/mcts_binding.cpp'],
            extra_compile_args={'cxx': ['-O3', '-std=c++17']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)