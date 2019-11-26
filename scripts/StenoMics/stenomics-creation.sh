#!/bin/bash

while [ -n "$1" ]; do # while loop starts
 
    case "$1" in
 
    --reset) 
        RESET_CONFIGS=true
        ;;

    --aec)
        AEC=true
        ;;

    --create-virtual-output)
        CREATE_VIRTUAL_OUTPUT=true
        ;;

    --default)
        USE_DEFAULT=true
        ;;

    *) echo "Option $1 not recognized" ;;
 
    esac
 
    shift
 
done

if $USE_DEFAULT; then
    DEFAULT_PA_PATH=~/.pulse/default.pa
    DAEMON_CONF_PATH=~/.pulse/daemon.conf
    CLIENT_CONF_PATH=~/.pulse/client.conf

    RESAMPLE_METHOD=speex-float-1

    #INPUT
    SAMPLE_RATE=16000
    SAMPLE_FORMAT=s32le
    SAMPLE_CHANNELS=16
    SOURCE_MASTER=alsa_input.usb-IntRoLab_16SoundsUSB_Audio_2.0-00.multichannel-input
    SINK_MASTER=alsa_output.usb-IntRoLab_16SoundsUSB_Audio_2.0-00.analog-surround-40
    INTROLAB_CARD_CHANNEL_MAP_8=front-left,front-right,rear-left,rear-right,front-center,lfe,side-left,side-right
    INTROLAB_CARD_CHANNEL_MAP_16=front-left,front-right,rear-left,rear-right,front-center,lfe,side-left,side-right,aux0,aux1,aux2,aux3,aux4,aux5,aux6,aux7

    #AEC
    AEC_METHOD=webrtc
    AEC_SOURCE_NAME=MicroFiltre
    AEC_SINK_NAME=SpeakerFiltre

    #OUTPUT
    SINK_NAME=webrtc_in
    ODAS_CHANNELS=4
    ODAS_CHANNEL_MAP=front-left,front-right,rear-left,rear-right
fi

if $RESET_CONFIGS; then
    rm -rf ~/.pulse
    mkdir ~/.pulse
    cp /etc/pulse/* ~/.pulse/

    sed -i "s/$(cat ~/.pulse/client.conf | grep autospawn)/autospawn = no/g" $CLIENT_CONF_PATH
fi

sed -i "s/$(cat $DAEMON_CONF_PATH | grep resample-method)/resample-method = $RESAMPLE_METHOD/g" "$DAEMON_CONF_PATH"
sed -i "s/$(cat $DAEMON_CONF_PATH | grep default-sample-format)/default-sample-format = $SAMPLE_FORMAT/g" "$DAEMON_CONF_PATH"
sed -i "s/$(cat $DAEMON_CONF_PATH | grep default-sample-rate)/default-sample-rate = $SAMPLE_RATE/g" "$DAEMON_CONF_PATH"
sed -i "s/$(cat $DAEMON_CONF_PATH | grep default-sample-channels)/default-sample-channels = $SAMPLE_CHANNELS/g" "$DAEMON_CONF_PATH"

if pgrep -x "pulseaudio" > /dev/null; then
    echo "killing running pulseaudio..."
    pulseaudio -k
    sleep 1
fi
echo "starting pulseaudio..."
pulseaudio --start --system
sleep 3

if $CREATE_VIRTUAL_OUTPUT || $USE_DEFAULT; then
    pacmd load-module module-null-sink sink_name=$SINK_NAME format=$SAMPLE_FORMAT rate=$SAMPLE_RATE channels=$ODAS_CHANNELS channel_map=$ODAS_CHANNEL_MAP
fi

if $AEC || $USE_DEFAULT; then
    if [[ "$SAMPLE_CHANNELS" -eq 8 ]]; then
        # 8 channels
        SOURCE_MASTER_8_CH=source_8_ch
        pacmd load-module module-remap-source master=$SOURCE_MASTER source_name=$SOURCE_MASTER_8_CH format=s32le rate=$SAMPLE_RATE channels=8 master_channel_map=$INTROLAB_CARD_CHANNEL_MAP_8 channel_map=$INTROLAB_CARD_CHANNEL_MAP_8
        pacmd load-module module-echo-cancel source_name=$AEC_SOURCE_NAME source_master=$SOURCE_MASTER_8_CH sink_name=$AEC_SINK_NAME sink_master=$SINK_MASTER aec_method=$AEC_METHOD use_master_format=1 aec_args="'high_pass_filter=1 noise_suppression=1 analog_gain_control=0'"
    else
        # 16 channels
        pacmd load-module module-echo-cancel source_name=$AEC_SOURCE_NAME source_master=$SOURCE_MASTER sink_name=$AEC_SINK_NAME sink_master=$SINK_MASTER aec_method=$AEC_METHOD use_master_format=1 aec_args="'high_pass_filter=1 noise_suppression=1 analog_gain_control=0'"
    fi

    pacmd set-default-source $AEC_SOURCE_NAME
    pacmd set-default-sink $AEC_SINK_NAME
fi

echo "done!"


