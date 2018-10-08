#!/bin/bash

# Cache directory for apt-get packages
mkdir -p ".vendor/apt"

apt-get update -yq

apt-get -o dir::cache::archives=".vendor/apt" install -yq wget build-essential pkg-config gfortran libatlas-base-dev python-dev python-pip libffi-dev libssl-dev
# Matplotlib dependencies
apt-get -o dir::cache::archives=".vendor/apt" install -yq python-tk libpng-dev libgif-dev libjpeg8-dev libtiff5-dev libpng12-dev libfreetype6-dev
# Update to latest pip version
pip install -U pip

exit 0