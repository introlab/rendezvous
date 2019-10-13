# Jetson setup for the Rendezvous project.

## OS
Install Jetpack 3.3 using the following guide:
    https://developer.nvidia.com/embedded/dlc/jetpack-install-guide-3_3

## Hardware: Install the CSI camera driver
This is only meant for users who already flashed the standard L4T_R28.2.1/L4T_R28.2 (Jetpack v3.3) package for TX2.
In this procedure, the existing kernel Image and the device tree blob (dtb) in the Jetson
board will be replaced by the pre-built binaries given with the Release package; Existing
Root Filesystem in the Jetson board will be preserved by this procedure. 

To complete this procedure, you will need the e-CAM131_CUTX2_JETSON_TX2_L4T_28.2.1_18-SEP-2018_R01.tar.gz 
found on the SC Card supplied with the camera kit or you can download it at 
this address: https://www.e-consystems.com/13mp-nvidia-jetson-tx2-camera-board.asp

Setup the environment in a terminal window in Jetson, copy the release package to the
board and extract it using the following commands from the Jetson board.

	$ mkdir top_dir/ -p 
	$ export TOP_DIR=<absolute path to>/top_dir  
	$ export RELEASE_PACK_DIR=<absolute path to>/top_dir/e-CAM131_CUTX2_JETSON_TX2_L4T_28.2.1_18-SEP-2018_R01 
	$ cp e-CAM131_CUTX2_JETSON_TX2_L4T_28.2.1_18-SEP-2018_R01.tar.gz $TOP_DIR/ 
	$ cd $TOP_DIR 
	$ tar -xaf e-CAM131_CUTX2_JETSON_TX2_L4T_28.2.1_18-SEP-2018_R01.tar.gz

Now replace the existing kernel Image with e-con’s pre-built Image and related driver
modules given with the Release package using the commands below:

	$ sudo cp $RELEASE_PACK_DIR/Kernel/Binaries/Image /boot/Image -f

Copy the e-CAM131_CUTX2 Camera driver and other driver modules given with the release
package using the commands below:
	 
	$ sudo tar xjpmf $RELEASE_PACK_DIR/Kernel/Binaries/kernel_supplements.tar.bz2 -C / 
	$ sudo cp $RELEASE_PACK_DIR/Rootfs/modules /etc/modules -f
	 
Flash the Signed DTB to necessary eMMC partition (mmcblk0p26).

	$ sudo dd if=$RELEASE_PACK_DIR/Kernel/Binaries/tegra186-quill-p3310-1000-c03-00-base_sigheader.dtb.encrypt of=/dev/mmcblk0p26 bs=1M

Now reboot the Jetson development kit. The Jetson TX2 board will now be running the latest
binaries. The module drivers for e-CAM131_CUTX2 will be loaded automatically in the Jetson during
booting. Check the presence of video node using the following command: 

	$ ls /dev/video0

You can test the camera with this command:

	$ gst-launch-1.0 v4l2src device=/dev/video0 ! "video/x-raw, format=(string)UYVY, width=(int)3840,height=(int)2160" ! nvvidconv ! "video/x-raw(memory:NVMM), format=(string)I420, width=(int)1920,height=(int)1080" ! nvoverlaysink overlay-w=1920 overlay-h=1080 sync=false
	
	
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

Run the script

    $ mkdir dev/lib
    $ chmod +x scripts/install_opencv34.sh
    $ ./scripts/install_opencv34.sh

Add a link to the opencv library in the project env :

	$ mv /usr/local/lib/python3.5/site-packages/cv2.cpython-35m-aarch64-linux-gnu.so /usr/local/lib/python3.5/dist-packages/cv2.so
	$ ln -s /usr/local/lib/python3.5/dist-packages/cv2.so /home/nvidia/dev/workspace/rendezvous/env/lib/python3.5/site-packages/cv2.so

### ODAS

Install the following dependencies:

    $ sudo apt-get install libfftw3-dev -y
    $ sudo apt-get install libconfig-dev -y
    $ sudo apt-get install libasound2-dev -y

Build Odas:

    $ cd ~/dev/lib
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
    $ cd YOLO3-4-Py
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


### V4l2loopback

Installation:
    $ cd /usr/src/linux-headers-4.4.38-tegra
    $ sudo make modules_prepare
    $ sudo apt-get install v4l2loopback-dkms -y

To create a v4l2loopback device:
    $ sudo modprobe v4l2loopback

To remove a v4l2loopback device:
    $ sudo rmmod v4l2loopback

### libv4l2cpp

    $ cd ~/dev/lib
    $ git clone https://github.com/mpromonet/libv4l2cpp
    $ chmod +x ../workspace/rendezvous/scripts/build-libv4l2cpp.sh
    $ ../workspace/rendezvous/scripts/build-libv4l2cpp.sh