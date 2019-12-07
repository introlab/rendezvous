#!/bin/bash

# Default values
INSTALL_PATH=$PWD

# Check for optional parameters
if ! [ -z "$1" ]; then
    INSTALL_PATH=$1
fi

# Clone ODAS and compile 
mkdir -p $INSTALL_PATH
cd $INSTALL_PATH
INSTALL_PATH=$PWD
git clone https://github.com/pbeaulieu26/odas.git
cd odas
mkdir build
cd build
cmake ../
make
