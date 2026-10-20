from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
setup(name='fast_liquid_calibration',ext_modules=cythonize([Extension('fast_liquid_calibration',['fast_liquid_calibration.pyx'],include_dirs=[numpy.get_include()],extra_compile_args=['/O2','/fp:precise'])],language_level=3))
