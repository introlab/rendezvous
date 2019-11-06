#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

#include "model/settings/base_config.h"

#include <string>

namespace Model
{
class AudioConfig : public BaseConfig
{
Q_OBJECT
public:
    enum Key
    {
        DEVICE_NAME,
        CHANNELS,
        RATE,
        FORMAT_BYTES,
        IS_LITTLE_ENDIAN,
        PACKET_AUDIO_SIZE,
        PACKET_HEADER_SIZE
    }; Q_ENUM(Key)

    AudioConfig(const QString &group, std::shared_ptr<QSettings> settings)
        : BaseConfig(group, settings)
    {
        update();
    }

    void update()
    {
        deviceName = value(Key::DEVICE_NAME).toString().toStdString();
        channels = value(Key::CHANNELS).toInt();
        rate = value(Key::RATE).toInt();
        formatBytes = value(Key::FORMAT_BYTES).toInt();
        isLittleEndian = value(Key::IS_LITTLE_ENDIAN).toBool();
        packetAudioSize = value(Key::PACKET_AUDIO_SIZE).toInt();
        packetHeaderSize = value(Key::PACKET_HEADER_SIZE).toInt();
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
