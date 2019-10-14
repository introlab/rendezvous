#!/bin/bash

REPO_DIR=$PWD

### Clone and compile darknet code ###
INSTALL_PATH=~/dev/lib
USE_GPU=1

if ! [ -z "$1" ]
  then

    # Check for installation path and if compilation uses GPU
    if [ "$1" == "CPU" ] || [ "$2" == "CPU" ]
      then
        USE_GPU=0
    fi
    if [ "$1" != "CPU" ]
      then
        INSTALL_PATH=$1
    fi

    # Remove '/' at the end if present
    if [ "${INSTALL_PATH: -1}" == "/" ]
      then
        INSTALL_PATH=${INSTALL_PATH::-1}
    fi
fi

mkdir -p $INSTALL_PATH
cd $INSTALL_PATH
git clone https://github.com/MathieuFavreau/darknet.git
cd darknet
git checkout origin/predict-gpu-device-image
make GPU=$USE_GPU ARCH="-gencode arch=compute_50,code=[sm_50,compute_50] -gencode arch=compute_53,code=[sm_53,compute_53] -gencode arch=compute_62,code=[sm_62,compute_62]"

cd $REPO_DIR

### Setup darknet environement paths ###

if grep -qF "export DARKNET_HOME" ~/.bashrc;then
   sed -i 's@export DARKNET_HOME=.*@export DARKNET_HOME='"$INSTALL_PATH"'/darknet@g' ~/.bashrc
else
   echo "" >> ~/.bashrc
   echo "export DARKNET_HOME=$INSTALL_PATH/darknet" >> ~/.bashrc
fi

if ! grep -qF "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$DARKNET_HOME" ~/.bashrc;then
   echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$DARKNET_HOME" >> ~/.bashrc
fi