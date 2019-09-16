# Jetson setup for the Rendezvous project.

## OS
Install Jetpack 3.3 using the following guide:
    https://developer.nvidia.com/embedded/dlc/jetpack-install-guide-3_3

## Project and dependencies
*Note: Keep the same console for the whole process and do everything in order.

### General dependencies
Run the following commands:

    $ sudo apt-get install python3 -y
    $ sudo apt-get install python3-dev -y
    $ sudo apt-get install cython
    $ sudo apt-get install python3-pip -y
    $ sudo pip3 install --upgrade pip

    $ sudo apt-get install cmake -y

    $ sudo apt-get install python3-tk -y
    $ sudo apt-get install xorg-dev libglu1-mesa-dev -y
    $ sudo apt-get install swig -y
    $ sudo apt-get install ffmpeg -y
    $ sudo apt-get install autoconf -y
    $ sudo apt-get install libtool -y

Create the directories:

    $ cd ~
    $ mkdir dev
    $ mkdir dev/workspace
    $ mkdir dev/lib

### RendezVous
Get the repo

    $ cd ~/dev/workspace
    $ git clone https://github.com/introlab/rendezvous.git

Setup and activate the environment

    $ cd rendezvous
    $ python3 -m pip install --user virtualenv
    $ python3 -m virtualenv env
    $ source env/bin/activate

Install the requirements:

    $ pip3 install -r requirements-jetson.txt

### OpenCV 3.4
    $ mkdir dev/lib
    $ chmod +x scripts/install_opencv34.sh
    $ ./scripts/install_opencv34.sh

In the file /usr/local/cuda/include cuda_gl_interop.h, modify the following lines (add the comments).

	//#if defined(__arm__) || defined(__aarch64__)
	//#ifndef GL_VERSION
	//#error Please include the appropriate gl headers before including cuda_gl_interop.h
	//#endif
	//#else
	 #include <GL/gl.h>
	//#endif

Create the following link:

    $ cd /usr/lib/aarch64-linux-gnu/
    $ sudo ln -sf tegra/libGL.so libGL.so

Add a link to the opencv library in the project env :

	$ mv /usr/local/lib/python3.5 dist-packages cv2.cpython-35m-aarch64-linux-gnu.so /usr/local/lib/python3.5/dist-packages/cv2.so
	$ ln -s /usr/local/lib/python3.5/dist-packages/cv2.so /home/nvidia/dev/workspace/rendezvous/env/lib/python3.5/site-packages/cv2.so

### ODAS

Install the following dependencies:

    $ sudo apt-get install libfftw3-dev -y
    $ sudo apt-get install libconfig-dev -y
    $ sudo apt-get install libasound2-dev -y

Build Odas:

    $ cd ~/dev/workspace
    $ git clone https://github.com/introlab/odas.git
    $ cd odas
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make



### PyQt5

Install the following dependencies:

    $ sudo apt-get install python3-pyqt5 -y
    $ sudo apt-get install pyqt5-dev-tools -y
    $ sudo apt-get install qttools5-dev-tools -y 
    $ sudo apt-get install python3-sip -y
    $ sudo apt-get install python3-scipy -y

Copy the following files in the project env:

    $ cp -r /usr/lib/python3/dist-packages/PyQt5 ~/dev/workspace/rendezvous/env/lib/python3.5/site-packages
    $ cp -r /usr/lib/python3/dist-packages/scipy ~/dev/workspace/rendezvous/env/lib/python3.5/site-packages
    $ cp /usr/lib/python3/dist-packages/sipconfig.py ~/dev/workspace/rendezvous/env/lib/python3.5/site-packages
    $ cp /usr/lib/python3/dist-packages/sipconfig_nd5.py ~/dev/workspace/rendezvous/env/lib/python3.5/site-packages
    $ cp /usr/lib/python3/dist-packages/sip.cpython-35m-aarch64-linux-gnu.so ~/dev/workspace/rendezvous/env/lib/python3.5/site-packages



### RNNoise

    $ cd ~/dev/lib
    $ git clone https://github.com/xiph/rnnoise
    $ cd rnnoise
    $ ./autogen.sh
    $ ./configure
    $ sudo make install



### Yolov3

Build Yolov3 from source:

    $ cd ~/dev/lib 
    $ git clone https://github.com/pjreddie/darknet yolov3
    $ cd yolov3

Change the following lines in the Makefile:

	GPU=1
	CUDNN=1
	OPENCV=1
	......
	ARCH= -gencode arch=compute_53,code=[sm_53,compute_53] \
	      -gencode arch=compute_62,code=[sm_62,compute_62]

Build:

    $ make

Add the following lines at the end of ~/.bashrc:

	export DARKNET_HOME=$HOME/dev/lib/yolov3
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DARKNET_HOME
	export CUDA_HOME=/usr/local/cuda

Install Yolov3 for python in the project env:

    $ cd ~/dev/workspace/rendezvous
    $ git clone https://github.com/madhawav/YOLO3-4-Py.git
    $ cd Yolo3-4-Py.git
    $ export CUDA=1
    $ export OPENCV=1 
    $ pip3 install .
    $ cd ..
    $ rm -rf Yolo3-4-Py.git

Download the neural network model and data:

    $ chmod +x scripts/yolo_setup.sh
    $ ./scripts/yolo_setup.sh



### VLC

    $ cd ~/dev/lib
    $ sudo apt-get build-dep vlc && sudo apt-get install libtool build-essential
    $ wget http://download.videolan.org/pub/videolan/vlc/2.2.6/vlc-2.2.6.tar.xz  
    $ tar -xf vlc-2.2.6.tar.xz 
    $ rm  vlc-2.2.6.tar.xz
    $ cd vlc-2.2.6
    $ ./configure --disable-qt
    $ sudo make install

    $ sudo apt-get install vlc-data=2.2.2-5ubuntu0.16.04.4
    $ sudo apt-get install libvlccore8
    $ sudo apt-get install libvlc5
    $ pip install python-vlc