from setuptools import setup, Extension
from torch.utils import cpp_extension

__version__ = "0.1.0"

setup(
    name='cppeff',
    version=__version__,
    ext_modules=[cpp_extension.CppExtension('cppeff', ['special_add.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
