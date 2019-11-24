#!/bin/bash

sudo cp $PWD/scripts/overclock-jetson/overclock-jetson.sh /usr/bin/overclock-jetson.sh
sudo chmod +x /usr/bin/overclock-jetson.sh

sudo cp $PWD/scripts/overclock-jetson/overclock-jetson.service /etc/systemd/system/overclock-jetson.service
sudo chmod 644 /etc/systemd/system/overclock-jetson.service

sudo systemctl enable overclock-jetson.service
