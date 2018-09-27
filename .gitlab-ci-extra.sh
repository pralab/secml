#!/bin/bash

apt_llvm_trusty()
{
  apt-get install -qq -y wget software-properties-common
  wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key|apt-key add -
  apt-add-repository "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-3.7 main"
  apt-get update -qq
}

# Ubuntu trusty reps don't have llvm-3.7 package. Need to get them from developers
apt-get update -qq
apt-get install -qq -y lsb-release
if [ "$(lsb_release -cs)" = "trusty" ]; then
  echo "Adding llvm repository for ubuntu:trusty"
  apt_llvm_trusty
fi

# Cache directory for apt-get packages
mkdir -p ".vendor/apt"

apt-get -o dir::cache::archives=".vendor/apt" install -y -qq wget build-essential pkg-config gfortran libatlas-base-dev python-dev python-pip libffi-dev libssl-dev
# Numba dependencies
apt-get -o dir::cache::archives=".vendor/apt" install -y -qq m4 zlib1g zlib1g-dev libedit2 libedit-dev llvm-3.7 llvm-3.7-dev llvm-dev
# Matplotlib dependencies
apt-get -o dir::cache::archives=".vendor/apt" install -y -qq python-tk libpng-dev libgif-dev libjpeg8-dev libtiff5-dev libpng12-dev libfreetype6-dev
# OpenOpt dependencies
apt-get -o dir::cache::archives=".vendor/apt" install -y -qq libgmp3-dev libglpk-dev python-cvxopt
# Update to latest pip version
pip install -U pip

# Installation of OpenOpt dependencies from source
# wget -q https://gmplib.org/download/gmp/gmp-6.1.0.tar.bz2 && tar -jxf gmp-6.1.0.tar.bz2
# wget -q http://ftp.gnu.org/gnu/glpk/glpk-4.59.tar.gz && tar -xzf glpk-4.59.tar.gz
# wget -q https://github.com/cvxopt/cvxopt/archive/1.1.8.tar.gz && tar -xzf 1.1.8.tar.gz
# cd ./gmp-6.1.0/ && ./configure && make > /dev/null && make check > /dev/null && make install > /dev/null
# cd ../glpk-4.59/ && ./configure --with-gmp && make > /dev/null && make check > /dev/null && make install > /dev/null
# cd ../cvxopt-1.1.8 && CVXOPT_BUILD_GLPK=1 CVXOPT_GLPK_LIB_DIR='/usr/local/lib' CVXOPT_GLPK_INC_DIR='/usr/local/include' python setup.py install > /dev/null && ldconfig

exit 0