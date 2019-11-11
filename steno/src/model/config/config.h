#ifndef SETTINGS_H
#define SETTINGS_H

#include "base_config.h"

#include <memory>

#include <QSettings>
#include <QString>

namespace Model
{
class AppConfig;
class AudioConfig;
class StreamConfig;
class DewarpingConfig;
class VideoConfig;
class TranscriptionConfig;
class DarknetConfig;

class Config : public BaseConfig
{
    Q_OBJECT
   public:
    enum Group
    {
        APP,
        TRANSCRIPTION,
        DEWARPING,
        VIDEO_INPUT,
        VIDEO_OUTPUT,
        AUDIO_INPUT,
        AUDIO_OUTPUT,
        STREAM,
        DARKNET
    };
    Q_ENUM(Group)

    Config(std::shared_ptr<QSettings> settings, const QString& configPath);

    std::shared_ptr<AppConfig> appConfig() const;
    std::shared_ptr<DewarpingConfig> dewarpingConfig() const;
    std::shared_ptr<VideoConfig> videoInputConfig() const;
    std::shared_ptr<VideoConfig> videoOutputConfig() const;
    std::shared_ptr<AudioConfig> audioInputConfig() const;
    std::shared_ptr<AudioConfig> audioOutputConfig() const;
    std::shared_ptr<StreamConfig> streamConfig() const;
    std::shared_ptr<TranscriptionConfig> transcriptionConfig() const;
    std::shared_ptr<DarknetConfig> darknetConfig() const;

   private:
    void loadDefault();

    std::shared_ptr<AppConfig> m_appConfig;
    std::shared_ptr<DewarpingConfig> m_dewarpingConfig;
    std::shared_ptr<VideoConfig> m_videoInputConfig;
    std::shared_ptr<VideoConfig> m_videoOutputConfig;
    std::shared_ptr<AudioConfig> m_audioInputConfig;
    std::shared_ptr<AudioConfig> m_audioOutputConfig;
    std::shared_ptr<StreamConfig> m_streamConfig;
    std::shared_ptr<TranscriptionConfig> m_transcriptionConfig;
    std::shared_ptr<DarknetConfig> m_darknetConfig;
};

}    // namespace Model

#endif    // SETTINGS_H
