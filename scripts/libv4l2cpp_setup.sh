#!/bin/bash

# Default values
INSTALL_PATH=$PWD

# Check for optional parameters
if ! [ -z "$1" ]; then
    INSTALL_PATH=$1
fi

# Clone libv4l2cpp and compile the lib
mkdir -p $INSTALL_PATH
cd $INSTALL_PATH
INSTALL_PATH=$PWD
git clone https://github.com/mpromonet/libv4l2cpp
cd libv4l2cpp
make EXTRA_CXXFLAGS='-fPIC'

