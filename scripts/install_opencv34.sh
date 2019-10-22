#!/usr/bin/env bash
# Automated script to install OpenCV 3.4.
# Tested on Ubuntu 16.04

cd ~/dev/lib
mkdir opencv
cd opencv

sudo apt-get update
sudo apt-get install -y build-essential cmake
sudo apt-get install -y qt5-default libvtk6-dev
sudo apt-get install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libjasper-dev libopenexr-dev libgdal-dev
sudo apt-get install -y libdc1394-22-dev libavcodec-dev libavformat-dev libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev libxine2-dev
sudo apt-get install -y python-dev python-tk python-numpy python3-dev python3-tk python3-numpy

sudo apt-get install -y unzip wget
wget https://github.com/opencv/opencv/archive/3.4.0.zip
unzip 3.4.0.zip
rm 3.4.0.zip

cd opencv-3.4.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_CUDA=ON -D CUDA_ARCH_BIN="6.2" -D CUDA_ARCH_PTX="" \
      -D WITH_CUBLAS=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON \
      -D ENABLE_NEON=ON -D WITH_LIBV4L=ON -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF \
      -D WITH_QT=ON -D WITH_OPENGL=ON ..
make -j4
sudo make install
sudo ldconfig
