#!/bin/bash

# Default values
INSTALL_PATH=~/dev/lib

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

# Setup libv4l2cpp environement paths in ~/.bashrc
if grep -qF "export LIBV4L2CPP_HOME" ~/.bashrc;then
    sed -i 's@export LIBV4L2CPP_HOME=.*@export LIBV4L2CPP_HOME='"$INSTALL_PATH"'/libv4l2cpp@g' ~/.bashrc
else
    echo "" >> ~/.bashrc
    echo "export LIBV4L2CPP_HOME=$INSTALL_PATH/libv4l2cpp" >> ~/.bashrc
fi