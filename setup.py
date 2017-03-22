from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules=[
    Extension("inpainting",
              sources=["inpainting.pyx"],
              libraries=["m"],
              extra_compile_args=["-O3", "-ffast-math", "-march=native", "-fopenmp"],
              extra_link_args=["-fopenmp"])
]

setup(
    name="inpainting",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules
)
