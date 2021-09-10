#!/usr/bin/python3

# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# from Cython.Build import cythonize
#
# import numpy as np
#
# cyarma_extension = Extension(
#     name='l0learn.l0learn.cyarma',
#     sources=["l0learn/l0learn/cyarma.pyx"],
#     include_dirs=['.', np.get_include()],
#     language="c++",
#     extra_compile_args=["-std=c++11"],
#     extra_link_args=["-std=c++11"],
# )
#
# utils_cyarma_extension = Extension(
#     name='l0learn.tests.utils_cyarma',
#     sources=["l0learn/tests/testing_utils.pyx"],
#     include_dirs=['.', np.get_include()],
#     language="c++",
#     extra_compile_args=["-std=c++11"],
#     extra_link_args=["-std=c++11"],
# )
#
# bindings_extension = Extension(
#     name="l0learn.l0learn.interface",
#     sources=["l0learn/l0learn/interface.pyx"],
#     include_dirs=['.', np.get_include()],
#     language="c++",
#     extra_compile_args=["-std=c++11"],
#     extra_link_args=["-std=c++11"],
# )
#
# ext_modules = [cyarma_extension, utils_cyarma_extension, bindings_extension]
#
# # setup(
# #   name='py_bindings',
# #   ext_modules=cythonize(ext_modules)
# # )
#
# setup(
#     name='l0learn',
#     packages=['l0learn'],
#     description='Python Wrapper for l0learn',
#     maintainer='Tim Nonet',
#     package_data={'l0learn': ['l0learn/l0learn/*.pyx', 'l0learn/l0learn/*.pxd']},
#     cmdclass={'build_ext': build_ext},
#     language_level=3,
#     ext_modules=cythonize(ext_modules),
# )

import os
from setuptools import setup, find_packages, Extension
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

try:
    from psutil import cpu_count

    psutil_found = True
except ImportError:
    psutil_found = False

from Cython.Distutils import build_ext


# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


if psutil_found:
    os.environ["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_count(logical=False))

extensions = [
    Extension(name='l0learn.cyarma',
              sources=["l0learn/cyarma.pyx"],
              include_dirs=['.', np.get_include()],  # "usr/local/Cellar/armadillo/10.1.1/include"
              language="c++",
              libraries=["armadillo", "lapack", "blas"],
              extra_compile_args=["-std=c++11"],
              extra_link_args=["-std=c++11"],
              # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              ),
    Extension(name='l0learn.testing_utils',
              sources=["l0learn/testing_utils.pyx"],
              include_dirs=['.', np.get_include()],
              language="c++",
              libraries=["armadillo", "lapack", "blas"],
              extra_compile_args=["-std=c++11"],
              extra_link_args=["-std=c++11"],
              # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
              ),
    Extension(name="l0learn.interface",
              sources=["l0learn/interface.pyx",
                       "l0learn/src/CDL012LogisticSwaps.cpp",
                       "l0learn/src/Grid2D.cpp",
                       "l0learn/src/CDL012SquaredHingeSwaps.cpp",
                       "l0learn/src/Normalize.cpp",
                       "l0learn/src/CDL012Swaps.cpp",
                       "l0learn/src/Grid.cpp",
                       "l0learn/src/Grid1D.cpp"
                       ],
              include_dirs=['.', np.get_include(), "l0learn/src/include"],
              language="c++",
              libraries=["armadillo", "lapack", "blas"],
              extra_compile_args=["-std=c++11"],
              extra_link_args=["-std=c++11"]),
    # Extension("l0learn_src",
    #           sources=[
    #               "l0learn/src/CDL012LogisticSwaps.cpp",
    #               "l0learn/src/Grid2D.cpp",
    #               "l0learn/src/CDL012SquaredHingeSwaps.cpp",
    #               "l0learn/src/Normalize.cpp",
    #               "l0learn/src/CDL012Swaps.cpp",
    #               "l0learn/src/Grid.cpp",
    #               "l0learn/src/Grid1D.cpp"
    #           ],
    #           include_dirs=["l0learn/src/include"],
    #           extra_compile_args=["-std=c++11"],
    #           extra_link_args=["-std=c++11"],
    #           )
]

CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None
if True:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    extensions = no_cythonize(extensions)

# with open("requirements.txt") as fp:
#     install_requires = fp.read().strip().split("\n")
#
# with open("requirements-dev.txt") as fp:
#     dev_requires = fp.read().strip().split("\n")

"""
Installation Notes;
How to ensure proper underlying armadillo is installed?
    MacOsX: brew install armadillo --with-hdf5
"""

install_requires = ['numpy', 'h5py']

setup(
    ext_modules=extensions,
    name='l0learn',
    description='Python Wrapper for L0Learn',
    maintainer='Tim Nonet',
    install_requires=['cython'],
    cmdclass={'build_ext': build_ext}
)

# setup(
#     ext_modules=extensions,
#     install_requires=install_requires,
#     extras_require={
#         "dev": dev_requires,
#         "docs": ["sphinx", "sphinx-rtd-theme"]
#     },
# )
