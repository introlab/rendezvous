#!/bin/bash

sudo chmod +x $PWD/scripts/StenoCam/stenocam-creation.sh
sudo cp $PWD/scripts/StenoCam/stenocam-creation.sh /usr/bin/stenocam-creation.sh
sudo chmod +x /usr/bin/stenocam-creation.sh

sudo cp $PWD/scripts/StenoCam/stenocam.service /etc/systemd/system/stenocam.service
sudo chmod 644 /etc/systemd/system/stenocam.service

sudo systemctl daemon-reload
sudo systemctl start stenocam.service && sudo systemctl enable stenocam.service
sudo systemctl daemon-reload
