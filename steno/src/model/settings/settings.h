#ifndef SETTINGS_H
#define SETTINGS_H

#include "model/app_config.h"
#include "model/settings/base_config.h"
#include "model/stream/audio/audio_config.h"
#include "model/stream/stream_config.h"
#include "model/stream/video/dewarping/models/dewarping_config.h"
#include "model/stream/video/video_config.h"
#include "model/transcription/transcription_config.h"
#include "settings_constants.h"

#include <memory>

#include <QSettings>
#include <QString>
#include <QVariant>

namespace Model
{
class Settings : public BaseConfig
{
    Q_OBJECT
   public:
    enum Group
    {
        GENERAL,
        TRANSCRIPTION,
        DEWARPING,
        VIDEO_INPUT,
        VIDEO_OUTPUT,
        AUDIO_INPUT,
        AUDIO_OUTPUT,
        STREAM
    };
    Q_ENUM(Group)

    Settings(std::shared_ptr<QSettings> settings);

    const AppConfig& appConfig() const;
    AppConfig& appConfig();
    const DewarpingConfig& dewarpingConfig() const;
    DewarpingConfig& dewarpingConfig();
    const VideoConfig& videoInputConfig() const;
    VideoConfig& videoInputConfig();
    const VideoConfig& videoOutputConfig() const;
    VideoConfig& videoOutputConfig();
    const AudioConfig& audioInputConfig() const;
    AudioConfig& audioInputConfig();
    const AudioConfig& audioOutputConfig() const;
    AudioConfig& audioOutputConfig();
    const StreamConfig& streamConfig() const;
    StreamConfig& streamConfig();
    const TranscriptionConfig& transcriptionConfig() const;
    TranscriptionConfig& transcriptionConfig();

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
};

}    // namespace Model

#endif    // SETTINGS_H
