#!/bin/bash

if [ -z ${PULSEAUDIO_HOME+x} ]
then
    echo "env variable PULSEAUDIO_HOME not set!"
    exit
fi

LOGS_DIRECTORY=$PULSEAUDIO_HOME/logs
if [ ! -d "$LOGS_DIRECTORY" ]; then
  mkdir -p $LOGS_DIRECTORY
fi

echo "Starting pulseaudio..."

$PULSEAUDIO_HOME/src/pulseaudio -n -F $PULSEAUDIO_HOME/src/default.pa -p $PULSEAUDIO_HOME/src/ -vvvv &> $LOGS_DIRECTORY/log.txt &
sleep 2

echo "Pulseaudio is running... logging output to $LOGS_DIRECTORY/log.txt"

# temporary for 2 speakers
pacmd load-module module-remap-sink sink_name=front_stereo master=alsa_output.usb-IntRoLab_16SoundsUSB_Audio_2.0-00.analog-surround-40 channels=2 master_channel_map=front-left,front-right channel_map=front-left,front-right remix=no

#temporary for 8 mics
pacmd load-module module-remap-source source_name=mic_stereo master=alsa_input.usb-IntRoLab_16SoundsUSB_Audio_2.0-00.multichannel-input channels=8

pacmd load-module module-echo-cancel source_name=micro_filtre source_master=mic_stereo sink_name=speaker_filtre sink_master=front_stereo use_master_format=1 aec_method='webrtc' aec_args='"high_pass_filter=1 noise_suppression=0 analog_gain_control=0"'

pacmd update-source-proplist micro_filtre device.description=MicroFiltre
pacmd update-sink-proplist speaker_filtre device.description=SpeakerFiltre

pacmd set-default-source micro_filtre
pacmd set-default-sink speaker_filtre

m_index=$(pactl list short sources | grep micro_filtre | awk '{print $1}')
s_index=$(pactl list short sinks | grep speaker_filtre | awk '{print $1}')

# Virtual mic for webrtc
pacmd load-module module-null-sink sink_name=webrtc_mic format=s16le rate=44100 channels=2

echo "Done!"
