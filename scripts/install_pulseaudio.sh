#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "run script with: pulseaudio.sh {absolute path for installing}"
    exit
fi

CURRENT_DIR=$(pwd)

INSTALL_DIR=$1

WEBRTC_HOME=$INSTALL_DIR/webrtc-audio-processing-0.2
PULSEAUDIO_HOME=$INSTALL_DIR/pulseaudio
PULSEAUDIO_BIN=$PULSEAUDIO_HOME/build

# Dependencies
sudo apt-get install autotools-dev \
                     autoconf \
                     libudev-dev \
                     libspeex-dev \
                     libspeexdsp-dev \
                     autopoint \
                     libsndfile-dev \
                     libcap-dev \
                     libdbus-1-dev \
                     intltool \
                     libasound2-dev

# WebRTC AEC library
# https://freedesktop.org/software/pulseaudio/webrtc-audio-processing/
#wget http://freedesktop.org/software/pulseaudio/webrtc-audio-processing/webrtc-audio-processing-0.2.tar.xz -O $INSTALL_DIR/webrtc-audio-processing-0.2.tar.xz
#tar -xf $INSTALL_DIR/webrtc-audio-processing-0.2.tar.xz -C $INSTALL_DIR
#cd $WEBRTC_HOME
#./configure
#make
#sudo make install
#sudo ldconfig

# Pulseaudio
# https://colin.guthr.ie/2010/09/compiling-and-running-pulseaudio-from-git/
# https://www.freedesktop.org/wiki/Software/PulseAudio/Documentation/Developer/PulseAudioFromGit/
git clone https://gitlab.freedesktop.org/pulseaudio/pulseaudio.git $INSTALL_DIR/pulseaudio
cd $PULSEAUDIO_HOME
git checkout tags/v11.1 -b v11.1
NOCONFIGURE=1 ./bootstrap.sh
mkdir build && cd build
CFLAGS="$CFLAGS -g -O0" ../configure --enable-force-preopen --enable-webrtc-aec --enable-udev --enable-dbus --enable-alsa --prefix=$(pwd)
mkdir -p src/daemon
make
mkdir -p share/pulseaudio && cd share/pulseaudio
ln -s ../../../src/modules/alsa/mixer 
cd ../../
ln -s pacat src/paplay
ln -s pacat src/parec

# Stop system pulse
mkdir -p ~/.pulse
echo "autospawn=no" >> ~/.pulse/client.conf

cd $CURRENT_DIR
