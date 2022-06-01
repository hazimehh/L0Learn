#!/bin/bash

# Based on https://github.com/mlpack/mlpack-wheels
# Based on https://gitlab.com/jason-rumengan/pyarma

basedir=$(python3 scripts/openblas_support.py)
$use_sudo cp -r $basedir/lib/* /usr/local/lib
$use_sudo cp $basedir/include/* /usr/local/include

export DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib/
