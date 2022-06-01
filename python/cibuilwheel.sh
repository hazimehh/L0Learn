python -m pip install cibuildwheel==2.5.0

export CIBW_SKIP="pp* *-win32 *-manylinux_i686 *musllinux*"
export CIBW_BEFORE_ALL_LINUX="yum install -y lapack-devel || apt-get -y liblapack-dev || apk add --update-cache --no-cache lapack-dev && bash scripts/install_linux_libs.sh"
export CIBW_BEFORE_TEST="pip install pytest numpy hypothesis"
export CIBW_TEST_COMMAND="python -u {package}/scripts/test_sparse.py"
export CIBW_BUILD_VERBOSITY=3
python -m cibuildwheel --output-dir wheelhouse --platform linux
