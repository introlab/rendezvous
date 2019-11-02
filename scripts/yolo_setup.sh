#!/bin/bash

# Get the root of the repo
REPO_DIR="`dirname \"$BASH_SOURCE\"`"
REPO_DIR="`( cd \"$REPO_DIR\" && pwd )`"
REPO_DIR="`dirname \"$REPO_DIR\"`"

cd $REPO_DIR

# Download neural network and data
git clone https://github.com/azmathmoosa/azFace

mkdir -p steno/configs/yolo/cfg
mkdir -p steno/configs/yolo/data
mkdir -p steno/configs/yolo/weights

cp azFace/net_cfg/tiny-yolo-azface-fddb.cfg steno/configs/yolo/cfg/
sed -i 's/width=416/width=608/g' steno/configs/yolo/cfg/tiny-yolo-azface-fddb.cfg
sed -i 's/height=416/height=608/g' steno/configs/yolo/cfg/tiny-yolo-azface-fddb.cfg

touch steno/configs/yolo/cfg/azface.data
echo "classes=1" >> steno/configs/yolo/cfg/azface.data
echo "names=$REPO_DIR/steno/configs/yolo/data/azface.names" >> steno/configs/yolo/cfg/azface.data

cp azFace/net_cfg/azface.names steno/configs/yolo/data/
cp azFace/weights/tiny-yolo-azface-fddb_82000.weights steno/configs/yolo/weights/

wget https://pjreddie.com/media/files/yolov3-tiny.weights --directory-prefix=steno/configs/yolo/weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg --directory-prefix=steno/configs/yolo/cfg

touch steno/configs/yolo/cfg/coco.data
echo "classes=1" >> steno/configs/yolo/cfg/coco.data
echo "names=$REPO_DIR/steno/configs/yolo/data/coco.names" >> steno/configs/yolo/cfg/coco.data

touch steno/configs/yolo/data/coco.names
echo "person" >> steno/configs/yolo/data/coco.names

rm -rf azFace
