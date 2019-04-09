#!/bin/bash

git clone https://github.com/azmathmoosa/azFace

mkdir -p config/yolo/cfg
mkdir -p config/yolo/data
mkdir -p config/yolo/weights

cp azFace/net_cfg/tiny-yolo-azface-fddb.cfg config/yolo/cfg/
sed -i 's/width=416/width=608/g' config/yolo/cfg/tiny-yolo-azface-fddb.cfg
sed -i 's/height=416/height=608/g' config/yolo/cfg/tiny-yolo-azface-fddb.cfg

cp azFace/net_cfg/azface.data config/yolo/cfg/
cp azFace/net_cfg/azface.names config/yolo/data/
cp azFace/weights/tiny-yolo-azface-fddb_82000.weights config/yolo/weights/

rm -rf azFace
