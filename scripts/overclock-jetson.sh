#!/bin/bash

sudo sh -c "echo 1 > /sys/devices/system/cpu/cpu1/online"
sudo sh -c "echo 1 > /sys/devices/system/cpu/cpu2/online"
sudo ~/jetson_clocks.sh
