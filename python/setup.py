#!/usr/bin/python3

import os

from setuptools import setup, Extension, find_packages
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

CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None
CYTHONIZE = True
COVERAGE_MODE = bool(os.getenv("L0LEARN_COVERAGE_MODE", 0)) and CYTHONIZE

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

if COVERAGE_MODE:
    macros = [("CYTHON_TRACE_NOGIL", "1")]
else:
    macros = []

extensions = [
    Extension(name='l0learn.cyarma',
              sources=["src/l0learn/cyarma.pyx"],
              include_dirs=['.', np.get_include()],
              language="c++",
              libraries=["armadillo", "lapack", "blas"],
              extra_compile_args=["-std=c++11"],
              extra_link_args=["-std=c++11"],
              define_macros=macros,
              ),
    Extension(name='l0learn.testing_utils',
              sources=["src/l0learn/testing_utils.pyx"],
              include_dirs=['.', np.get_include()],
              language="c++",
              libraries=["armadillo", "lapack", "blas"],
              extra_compile_args=["-std=c++11"],
              extra_link_args=["-std=c++11"],
              define_macros=macros,
              ),
    Extension(name="l0learn.interface",
              sources=["src/l0learn/interface.pyx",
                       "src/l0learn/src/CDL012LogisticSwaps.cpp",
                       "src/l0learn/src/Grid2D.cpp",
                       "src/l0learn/src/CDL012SquaredHingeSwaps.cpp",
                       "src/l0learn/src/Normalize.cpp",
                       "src/l0learn/src/CDL012Swaps.cpp",
                       "src/l0learn/src/Grid.cpp",
                       "src/l0learn/src/Grid1D.cpp"
                       ],
              include_dirs=['.', np.get_include(), "src/l0learn/src/include"],
              language="c++",
              libraries=["armadillo", "lapack", "blas"],
              extra_compile_args=["-std=c++11"],
              extra_link_args=["-std=c++11"],
              define_macros=macros,
              ),
]

if CYTHONIZE:
    compiler_directives = {"language_level": 3, "embedsignature": True}
    if COVERAGE_MODE:
        compiler_directives['linetrace'] = True
    extensions = cythonize(extensions, compiler_directives=compiler_directives)
else:
    COVERAGE_MODE = False
    extensions = no_cythonize(extensions)

"""
Installation Notes;
How to ensure proper underlying armadillo is installed?
    MacOsX: brew install armadillo --with-hdf5
"""

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='l0learn',
    maintainer='Tim Nonet',
    author_email="tim.nonet@gmail.com",
    description="L0Learn is a highly efficient framework for solving L0-regularized learning problems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hazimehh/L0Learn",
    project_urls={
        "Bug Tracker": "https://github.com/hazimehh/L0Learn/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    cmdclass={'build_ext': build_ext},
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    ext_modules=extensions,
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.1.0",
        "pandas>=1.0.0",
        "matplotlib>=3.0.0"
    ],
    extras_require={"test": [
        "attrs>=19.2.0",  # Usually installed by hypothesis, but current issue
        # #https://github.com/HypothesisWorks/hypothesis/issues/2113
        "hypothesis",
        "pytest",
    ]},
    python_requires=">=3.7",
)


