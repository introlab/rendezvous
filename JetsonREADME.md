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

Automation of pulse audio mics creation:

    sudo ./scripts/StenoMics/automation.sh

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

Compile darknet library:

    $ chmod +x scripts/darknet_setup.sh
    $ ./scripts/darknet_setup.sh

> Note: To compile without CUDA use -CPU

    $ ./scripts/darknet_setup.sh -CPU

> Note: To compile to a specific path

    $ ./scripts/darknet_setup.sh /my/path

Download the neural network model and data:

    chmod +x scripts/yolo_setup.sh
    ./scripts/yolo_setup.sh


## V4l2loopback

Installation

    sudo apt-get install v4l2loopback-dkms -y

To create a v4l2loopback device:
    
    sudo modprobe v4l2loopback

To remove a v4l2loopback device:
    
    sudo rmmod v4l2loopback

## libv4l2cpp

Compile libv4l2cpp library:

    $ chmod +x scripts/libv4l2cpp_setup.sh
    $ ./scripts/libv4l2cpp_setup.sh

> Note: To compile to a specific path

    $ ./scripts/libv4l2cpp_setup.sh /my/path

## Note

If you are missing some audio/video codecs, try the following command:

    $ sudo apt-get install ubuntu-restricted-extras

## References  

- https://doc.qt.io/qt-5/linux-building.html
- https://doc.qt.io/qt-5/linux-requirements.html
