#ifndef DARKNET_CONFIG_H
#define DARKNET_CONFIG_H

#include "model/config/base_config.h"

namespace Model
{
class DarknetConfig : public BaseConfig
{
    Q_OBJECT
   public:
    enum Key
    {
        SLEEP_BETWEEN_LAYERS_FORWARD_US
    };
    Q_ENUM(Key)

    DarknetConfig(const QString &group, std::shared_ptr<QSettings> settings)
        : BaseConfig(group, settings)
    {
        update();
    }

    void update()
    {
        sleepBetweenLayersForwardUs = value(Key::SLEEP_BETWEEN_LAYERS_FORWARD_US).toInt();
    }

    int sleepBetweenLayersForwardUs;
};
}    // namespace Model

#endif    // DARKNET_CONFIG_H
