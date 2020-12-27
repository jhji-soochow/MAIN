from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='deform_conv_cuda',
    ext_modules=[
        CUDAExtension(
                name='deform_conv_cuda',
                sources=['src/deform_conv_cuda.cpp', 'src/deform_conv_cuda_kernel.cu'],
                # extra_compile_args={'cxx': ['-g'],
                #                     'nvcc': ['-O2']}
                                    )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })