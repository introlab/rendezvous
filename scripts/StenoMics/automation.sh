#!/bin/bash

sudo chmod +x ~/dev/workspace/rendezvous/scripts/StenoMics/stenomics-creation.sh
sudo cp ~/dev/workspace/rendezvous/scripts/StenoMics/stenomics-creation.sh /usr/bin/stenomics-creation.sh
sudo chmod +x /usr/bin/stenomics-creation.sh

sudo cp ~/dev/workspace/rendezvous/scripts/StenoMics/stenomics.service /etc/systemd/system/stenomics.service
sudo chmod 644 /etc/systemd/system/stenomics.service

sudo systemctl daemon-reload
sudo systemctl start stenomics.service && sudo systemctl enable stenomics.service
sudo systemctl daemon-reload 