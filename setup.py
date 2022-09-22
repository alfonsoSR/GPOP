from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

# Extensions' configuration

library_dirs = ["/usr/local/lib"]

extra_compile_args = [
    "-ffast-math",
    "-march=native",
    "-msse2"
]

include_dirs = [
    "/usr/local/include",
    numpy.get_include()
]

# Extension modules

packages = [
    "gpop",
    "gpop.core",
    "gpop.pck",
    "gpop.utils",
    "gpop.old_solver"
]

ext_modules = [
    Extension(
        name="gpop.core.equations",
        sources=["src/gpop/core/equations.pyx"],
        libraries=["cspice", "m"],
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args
    ),
    "src/**/*.pyx"
]

if __name__ == "__main__":

    setup(
        name="gpop",
        include_dirs=include_dirs,
        packages=packages,
        ext_modules=cythonize(ext_modules, annotate=True)
    )
