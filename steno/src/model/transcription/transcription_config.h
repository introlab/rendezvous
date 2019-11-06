#ifndef TRANSCRIPTION_CONFIG_H
#define TRANSCRIPTION_CONFIG_H

#include "model/settings/base_config.h"

namespace Model
{
class TranscriptionConfig : public BaseConfig
{
    Q_OBJECT
   public:
    enum Key
    {
        LANGUAGE,
        AUTOMATIC_TRANSCRIPTION
    };
    Q_ENUM(Key)

    TranscriptionConfig(const QString &group, std::shared_ptr<QSettings> settings)
        : BaseConfig(group, settings)
    {
    }
};

}    // namespace Model

#endif    // TRANSCRIPTION_CONFIG_H
