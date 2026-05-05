"""
Build the axonforge_ops CUDA extension.

Run on a CUDA-capable machine (RunPod / AWS EC2):
  python setup.py install
  # or: pip install -e .

ptxas output (register count + SMEM) is printed at compile time via
--ptxas-options=-v. Redirect stderr to capture it:
  python setup.py install 2> ../results/cuda_op_ptxas.log
"""
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="axonforge_ops",
    ext_modules=[
        CUDAExtension(
            name="axonforge_ops",
            sources=[
                "swiglu_op.cpp",
                "kernels/swiglu_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-march=native"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_75,code=sm_75",
                    "--ptxas-options=-v",   # prints register + SMEM at compile
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
