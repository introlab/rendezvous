# Jetson setup for the Rendezvous project.

## Jetpack
Download and install the SDK Manager for the JetPack 4.2.2 on your host: 
    https://developer.nvidia.com/embedded/jetpack-archive

Follow the guide to configure the Jetson via the SDK Manager:
    https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html

## Overclock

https://github.com/introlab/rendezvous/wiki/Jetson-Overclock-Automation 

## Environnement

Create working directories

    cd ~
    mkdir dev
    mkdir dev/workspace
    mkdir dev/lib

Get the repository

    cd ~/dev/workspace
    git clone https://github.com/introlab/rendezvous.git

## Qt

### Requirements
Install platform plugin dependencies

    sudo apt-get install libfontconfig1-dev \
                         libfreetype6-dev \
                         libx11-dev \
                         libxext-dev \
                         libxfixes-dev \
                         libxi-dev \
                         libxrender-dev \
                         libxcb1-dev \
                         libx11-xcb-dev \
                         libxcb-glx0-dev \
                         libxkbcommon-x11-dev

Install multimedia dependencies

    sudoapt-get install libgstreamer1.0-0 \
                        gstreamer1.0-plugins-base \
                        gstreamer1.0-plugins-good \
                        gstreamer1.0-plugins-bad \
                        gstreamer1.0-plugins-ugly \
                        gstreamer1.0-libav \
                        gstreamer1.0-doc \
                        gstreamer1.0-tools \
                        gstreamer1.0-x \
                        gstreamer1.0-alsa \
                        gstreamer1.0-gl \
                        gstreamer1.0-gtk3 \
                        gstreamer1.0-qt5 \
                        gstreamer1.0-pulseaudio \
                        gstreamer0.10-ffmpeg

### Download and unpacking the archive
Download qt-everywhere 5.12.5 : https://download.qt.io/archive/qt/5.12/5.12.5/single/qt-everywhere-src-5.12.5.tar.xz 

Move the download to dev/lib/ and unpack

    mv $HOME/Downloads/qt-everywhere-src-5.12.5.tar.xz dev/lib
    cd ~/dev/lib/
    tar xvf qt-everywhere-src-5.12.5.tar.xz

### Build the library

Configure Qt build

    cd qt-everywhere-src-5.12.5
    ./configure -prefix /opt/qt5 \
                -confirm-license \
                -opensource \
                -nomake examples \
	            -nomake tests \
                -skip qtwebengine

Create and install Qt (~2-3 hours)

    make -j6
    make install

## Pulse Audio

Install the following dependencies:

    sudo apt-get install libpulse-dev   

## ODAS

Install the following dependencies:

    sudo apt-get install libfftw3-dev libconfig-dev libasound2-dev

Build Odas:

    cd ~/dev/lib
    git clone https://github.com/introlab/odas.git
    cd odas
    mkdir build
    cd build
    cmake ../
    make

## Yolov3

Build Yolov3 from source

    cd ~/dev/lib 
    git clone https://github.com/pjreddie/darknet yolov3
    cd yolov3

Change the following lines in the Makefile

	GPU=1
	CUDNN=1
	OPENCV=1
	......
	ARCH= -gencode arch=compute_53,code=[sm_53,compute_53] \
	      -gencode arch=compute_62,code=[sm_62,compute_62]

Build

    make

Add the following lines at the end of ~/.bashrc

	export DARKNET_HOME=$HOME/dev/lib/yolov3
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DARKNET_HOME
	export CUDA_HOME=/usr/local/cuda

Install Yolov3 for python in the project env

    cd ~/dev/workspace/rendezvous
    git clone https://github.com/madhawav/YOLO3-4-Py.git
    cd YOLO3-4-Py
    export CUDA=1
    export OPENCV=1 
    pip3 install .
    cd ..
    rm -rf Yolo3-4-Py.git

Download the neural network model and data

    chmod +x scripts/yolo_setup.sh
    ./scripts/yolo_setup.sh


## V4l2loopback

Installation

    cd /usr/src/linux-headers-4.4.38-tegra
    sudo make modules_prepare
    sudo apt-get install v4l2loopback-dkms -y

To create a v4l2loopback device:
    
    sudo modprobe v4l2loopback

To remove a v4l2loopback device:
    
    sudo rmmod v4l2loopback

## libv4l2cpp

Installation

    cd ~/dev/lib
    git clone https://github.com/mpromonet/libv4l2cpp
    cd libv4l2cpp/
    make EXTRA_CXXFLAGS='-fPIC'

Add the following line at the end of ~/.bashrc:

    export LIBV4L2CPP_HOME=$HOME/dev/lib/libv4l2cpp

## References  

- https://doc.qt.io/qt-5/linux-building.html
- https://doc.qt.io/qt-5/linux-requirements.html
