"""
Build the axonforge_allocator CUDA extension.

Run on a CUDA-capable machine:
  python setup_allocator.py install

The extension doubles as a CUDAPluggableAllocator shared library:
  axonforge_allocator.enable() registers it with PyTorch via
  torch.cuda.memory.CUDAPluggableAllocator pointing at the compiled .so.
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="axonforge_allocator",
    ext_modules=[
        CUDAExtension(
            name="axonforge_allocator",
            sources=["cuda_pool.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-march=native"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
