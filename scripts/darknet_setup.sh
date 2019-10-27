#!/bin/bash

###  Usage: ./setup_darknet.sh [install_path] [CPU]  ###
### [optional] install_path: folder of installation  ###
### [optional] CPU: Compile without GPU acceleration ###

# Default values
INSTALL_PATH=~/dev/lib
USE_GPU=1

params=($1 $2)

# Check for optional parameters
for param in "${params[@]}"
do
    if [ "$param" == "-CPU" ]; then
        USE_GPU=0
    elif ! [ -z "$param" ]; then
        INSTALL_PATH=$param
    fi
done

# Clone darknet and compile the lib
mkdir -p $INSTALL_PATH
cd $INSTALL_PATH
INSTALL_PATH=$PWD
git clone https://github.com/MathieuFavreau/darknet.git
cd darknet
git checkout origin/predict-gpu-device-image
make GPU=$USE_GPU ARCH="-gencode arch=compute_50,code=[sm_50,compute_50] -gencode arch=compute_53,code=[sm_53,compute_53] -gencode arch=compute_62,code=[sm_62,compute_62]"

# Setup darknet environement paths in ~/.bashrc
if grep -qF "export DARKNET_HOME" ~/.bashrc;then
    sed -i 's@export DARKNET_HOME=.*@export DARKNET_HOME='"$INSTALL_PATH"'/darknet@g' ~/.bashrc
else
    echo "" >> ~/.bashrc
    echo "export DARKNET_HOME=$INSTALL_PATH/darknet" >> ~/.bashrc
fi

if ! grep -qF "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$DARKNET_HOME" ~/.bashrc;then
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$DARKNET_HOME" >> ~/.bashrc
fi