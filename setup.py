from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

ext_modules=[
    Extension("inpainting",
              sources=["inpainting.pyx", "utils.c"],
              include_dirs=["numpy.get_include()"],
              language="c",
              libraries=["m"],
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=["-fopenmp"])
]

setup(
    name="inpainting",
    include_dirs=[numpy.get_include()],
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)
