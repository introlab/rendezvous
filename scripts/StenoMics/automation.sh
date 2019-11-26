#!/bin/bash

sudo chmod +x $PWD/scripts/StenoMics/stenomics-creation.sh
sudo cp $PWD/scripts/StenoMics/stenomics-creation.sh /usr/bin/stenomics-creation.sh
sudo chmod +x /usr/bin/stenomics-creation.sh

sudo cp $PWD/scripts/StenoMics/stenomics.service /etc/systemd/user/stenomics.service
sudo chmod 644 /etc/systemd/user/stenomics.service

systemctl --user daemon-reload
systemctl --user start stenomics.service && sudo systemctl --user enable stenomics.service
systemctl --user daemon-reload 