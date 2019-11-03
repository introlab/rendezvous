#!/bin/bash

# Get the root of the repo
REPO_DIR="`dirname \"$BASH_SOURCE\"`"
REPO_DIR="`( cd \"$REPO_DIR\" && pwd )`"
REPO_DIR="`dirname \"$REPO_DIR\"`"

cd $REPO_DIR

# Create the yolo config folders
mkdir -p steno/configs/yolo/cfg
mkdir -p steno/configs/yolo/data
mkdir -p steno/configs/yolo/weights

wget https://pjreddie.com/media/files/yolov3-tiny.weights --directory-prefix=steno/configs/yolo/weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg --directory-prefix=steno/configs/yolo/cfg

touch steno/configs/yolo/cfg/coco.data
echo "classes=1" >> steno/configs/yolo/cfg/coco.data
echo "names=$REPO_DIR/steno/configs/yolo/data/coco.names" >> steno/configs/yolo/cfg/coco.data

touch steno/configs/yolo/data/coco.names
echo "person" >> steno/configs/yolo/data/coco.names
