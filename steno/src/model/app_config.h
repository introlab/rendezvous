#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "config/base_config.h"

namespace Model
{
class AppConfig : public BaseConfig
{
    Q_OBJECT
   public:
    enum Key
    {
        OUTPUT_FOLDER,
        MICROPHONE_CONFIGURATION,
        ODAS_LIBRARY
    };
    Q_ENUM(Key)

    AppConfig(const QString &group, std::shared_ptr<QSettings> settings)
        : BaseConfig(group, settings)
    {
    }
};

}    // namespace Model

#endif    // APP_CONFIG_H
