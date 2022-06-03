#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# This was borrowed heavily form https://github.com/RUrlus/diptest/
import os
import sys

try:
    from skbuild import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from setuptools import find_packages
import io
import re
from os.path import dirname
from os.path import join
from distutils.sysconfig import get_python_inc, get_config_var

PACKAGE_NAME = 'l0learn'

MAJOR = 0
MINOR = 1
MICRO = 0
DEVELOPMENT = False

VERSION = f'{MAJOR}.{MINOR}.{MICRO}'
FULL_VERSION = VERSION
if DEVELOPMENT:
    FULL_VERSION += '.dev'


def read(*names, **kwargs):
    with io.open(
            join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setuptools_scm_conf = {"root": "..", "relative_to": __file__}
if os.getenv('SETUPTOOLS_SCM_NO_LOCAL', 'no') != 'no':
    setuptools_scm_conf['local_scheme'] = 'no-local-version'

if __name__ == '__main__':
    setup(
        name=PACKAGE_NAME,
        use_scm_version=setuptools_scm_conf,
        long_description_content_type="text/markdown",
        long_description="%s\n%s"
                         % (
                             re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
                                 "", read("README.md")
                             ),
                             re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.md")),
                         ),
        packages=find_packages(),
        setup_requires=["setuptools",
                        "wheel",
                        "scikit-build",
                        "cmake",
                        'setuptools_scm',
                        "ninja"],
        cmake_install_dir="l0learn",
        cmake_args=[
            f"-DL0LEARN_VERSION_INFO:STRING={VERSION}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPYTHON3_INCLUDE_DIR:STRING={get_python_inc()}",
            f"-DPYTHON3_LIBRARY:STRING={get_config_var('LIBDIR')}"
        ]
    )
