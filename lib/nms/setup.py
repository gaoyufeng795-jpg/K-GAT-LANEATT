# from setuptools import setup
#
# from torch.utils.cpp_extension import CUDAExtension, BuildExtension
#
# setup(name='nms', packages=['nms'],
#         package_dir={'':'src'},
#         ext_modules=[CUDAExtension('nms.details', ['src/nms.cpp', 'src/nms_kernel.cu'])],
#         cmdclass={'build_ext': BuildExtension})
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

extra_compile_args = {
    # 给 cl.exe：打开优化 + 允许“编译器和STL版本不匹配”
    'cxx': [
        '/O2',
        '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
    ],
    # 给 nvcc：忽略“不支持的编译器” + 把上面的宏转传给 host 编译器
    'nvcc': [
        '-O2',
        '-allow-unsupported-compiler',
        '-Xcompiler',
        '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
    ],
}

setup(
    name='nms',
    packages=['nms'],
    package_dir={'': 'src'},
    ext_modules=[
        CUDAExtension(
            'nms.details',
            ['src/nms.cpp', 'src/nms_kernel.cu'],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
