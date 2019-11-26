#!/bin/bash

sudo chmod +x /etc/transcription-api/scripts/prod.sh

sudo cp /etc/transcription-api/scripts/transcriptionAPI.service /etc/systemd/system/transcriptionAPI.service
sudo chmod 644 /etc/systemd/system/transcriptionAPI.service

sudo systemctl daemon-reload
sudo systemctl start transcriptionAPI.service && sudo systemctl enable transcriptionAPI.service
sudo systemctl daemon-reload