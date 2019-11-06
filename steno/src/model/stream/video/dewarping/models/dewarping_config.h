#ifndef DEWAPING_CONFIG_H
#define DEWAPING_CONFIG_H

#include "model/config/base_config.h"
#include "model/stream/utils/math/angle_calculations.h"

namespace Model
{
class DewarpingConfig : public BaseConfig
{
    Q_OBJECT
   public:
    enum Key
    {
        IN_RADIUS,
        OUT_RADIUS,
        ANGLE_SPAN,
        TOP_DISTORSION_FACTOR,
        BOTTOM_DISTORSION_FACTOR,
        FISH_EYE_ANGLE
    };
    Q_ENUM(Key)

    DewarpingConfig(const QString &group, std::shared_ptr<QSettings> settings)
        : BaseConfig(group, settings)
    {
        update();
    }

    void update()
    {
        inRadius = BaseConfig::value(Key::IN_RADIUS).toFloat();
        outRadius = value(Key::OUT_RADIUS).toFloat();
        angleSpan = math::deg2rad(value(Key::ANGLE_SPAN).toFloat());
        topDistorsionFactor = value(Key::TOP_DISTORSION_FACTOR).toFloat();
        bottomDistorsionFactor = value(Key::BOTTOM_DISTORSION_FACTOR).toFloat();
        fisheyeAngle = math::deg2rad(value(Key::FISH_EYE_ANGLE).toFloat());
    }

    float inRadius;
    float outRadius;
    float angleSpan;
    float topDistorsionFactor;
    float bottomDistorsionFactor;
    float fisheyeAngle;
};

}    // namespace Model

#endif    // !DEWAPING_CONFIG_H
