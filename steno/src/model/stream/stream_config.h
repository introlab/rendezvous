#ifndef STREAM_CONFIG_H
#define STREAM_CONFIG_H

#include "model/settings/base_config.h"
#include "model/stream/utils/math/angle_calculations.h"

namespace Model
{
class StreamConfig : public BaseConfig
{
Q_OBJECT
public:
    enum Key
    {
        DETECTION_DEWARPING_COUNT,
        ASPECT_RATIO_WIDTH,
        ASPECT_RATIO_HEIGHT,
        MIN_ELEVATION,
        MAX_ELEVATION
    }; Q_ENUM(Key)

    StreamConfig(const QString &group, std::shared_ptr<QSettings> settings)
        : BaseConfig(group, settings)
    {
        update();
    }

    void update()
    {
        detectionDewarpingCount = value(Key::DETECTION_DEWARPING_COUNT).toInt();
        aspectRatioWidth = value(Key::ASPECT_RATIO_WIDTH).toFloat();
        aspectRatioHeight = value(Key::ASPECT_RATIO_HEIGHT).toFloat();
        minElevation = math::deg2rad(value(Key::MIN_ELEVATION).toFloat());
        maxElevation = math::deg2rad(value(Key::MAX_ELEVATION).toFloat());
    }

    int detectionDewarpingCount;
    float aspectRatioWidth;
    float aspectRatioHeight;
    float minElevation;
    float maxElevation;
};
}

#endif // STREAM_CONFIG_H
