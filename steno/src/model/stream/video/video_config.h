#ifndef VIDEO_CONFIG_H
#define VIDEO_CONFIG_H

#include <string>

#include "model/settings/base_config.h"
#include "model/stream/utils/images/image_format.h"
#include "model/stream/utils/models/dim2.h"

namespace Model
{

class VideoConfig : public BaseConfig
{
Q_OBJECT
public:
    enum Key
    {
        FPS,
        WIDTH,
        HEIGHT,
        DEVICE_NAME,
        IMAGE_FORMAT
    }; Q_ENUM(Key)

    VideoConfig(const QString &group, std::shared_ptr<QSettings> settings)
        : BaseConfig(group, settings)
    {
        update();
    }

    void update()
    {
        resolution = Dim2<int>(value(Key::WIDTH).toInt(), value(Key::HEIGHT).toInt());
        fpsTarget = value(Key::FPS).toInt();
        deviceName = value(Key::DEVICE_NAME).toString().toStdString();
        imageFormat = static_cast<ImageFormat>(value(Key::IMAGE_FORMAT).toInt());
    }

    Dim2<int> resolution;
    int fpsTarget;
    std::string deviceName;
    ImageFormat imageFormat;
};

}    // namespace Model

#endif    // !VIDEO_CONFIG_H
