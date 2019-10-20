#!/bin/bash

# Get the root of the repo
REPO_DIR="`dirname \"$BASH_SOURCE\"`"
REPO_DIR="`( cd \"$REPO_DIR\" && pwd )`"
REPO_DIR="`dirname \"$REPO_DIR\"`"

cd $REPO_DIR

# Download neural network and data
git clone https://github.com/azmathmoosa/azFace

mkdir -p config/yolo/cfg
mkdir -p config/yolo/data
mkdir -p config/yolo/weights

cp azFace/net_cfg/tiny-yolo-azface-fddb.cfg config/yolo/cfg/
sed -i 's/width=416/width=608/g' config/yolo/cfg/tiny-yolo-azface-fddb.cfg
sed -i 's/height=416/height=608/g' config/yolo/cfg/tiny-yolo-azface-fddb.cfg

touch config/yolo/cfg/azface.data
echo "classes=1" >> config/yolo/cfg/azface.data
echo "names=$REPO_DIR/config/yolo/data/azface.names" >> config/yolo/cfg/azface.data

cp azFace/net_cfg/azface.names config/yolo/data/
cp azFace/weights/tiny-yolo-azface-fddb_82000.weights config/yolo/weights/

wget https://pjreddie.com/media/files/yolov3-tiny.weights --directory-prefix=config/yolo/weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg --directory-prefix=config/yolo/cfg

touch config/yolo/cfg/coco.data
echo "classes=1" >> config/yolo/cfg/coco.data
echo "names=$REPO_DIR/config/yolo/data/coco.names" >> config/yolo/cfg/coco.data

touch config/yolo/data/coco.names
echo "person" >> config/yolo/data/coco.names

rm -rf azFace
