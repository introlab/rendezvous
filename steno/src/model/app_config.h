#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include "model/settings/base_config.h"

namespace Model
{
class AppConfig : public BaseConfig
{
Q_OBJECT
public:
    enum Key
    {
        OUTPUT_FOLDER
    }; Q_ENUM(Key)

    AppConfig(const QString &group, std::shared_ptr<QSettings> settings)
        : BaseConfig(group, settings)
    {
    }
};

}    // namespace Model

#endif // APP_CONFIG_H
