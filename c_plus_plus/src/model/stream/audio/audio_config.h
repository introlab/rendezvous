#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

#include <string>

#include "model/stream/utils/images/image_format.h"
#include "model/stream/utils/models/dim2.h"

namespace Model
{
struct AudioConfig
{
    AudioConfig(std::string deviceName, int channels, int rate, int formatBytes, bool isLittleEndian, int packetAudioSize, int packetHeaderSize)
        : deviceName(std::move(deviceName))
        , channels(channels)
        , rate(rate)
        , formatBytes(formatBytes)
        , isLittleEndian(isLittleEndian)
        , packetAudioSize(packetAudioSize)
        , packetHeaderSize(packetHeaderSize)
    {
    }

    std::string deviceName;
    int channels;
    int rate;
    int formatBytes;
    bool isLittleEndian;
    int packetAudioSize;
    int packetHeaderSize;
};

}    // namespace Model

#endif    // !AUDIO_CONFIG_H
