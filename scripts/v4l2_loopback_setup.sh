#!/bin/bash

# Default values
INSTALL_PATH=$PWD

# Check for optional parameters
if ! [ -z "$1" ]; then
    INSTALL_PATH=$1
fi

# Clone v4l2loopback and compile
mkdir -p $INSTALL_PATH
cd $INSTALL_PATH
INSTALL_PATH=$PWD
git clone https://github.com/umlaeute/v4l2loopback.git
cd v4l2loopback
make
make install
depmod -a

cd ..
chmod +x ./scripts/StenoCam/stenocam-creation.sh
chmod +x ./scripts/StenoCam/automation.sh
./scripts/StenoCam/automation.sh
