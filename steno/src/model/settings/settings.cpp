#include "settings.h"

#include "model/transcription/transcription_constants.h"
#include "model/stream/utils/images/image_format.h"

#include <QFileInfo>

namespace Model
{
Settings::Settings(std::shared_ptr<QSettings> settings)
    : BaseConfig(settings)
    , m_appConfig(std::make_shared<Model::AppConfig>(QVariant::fromValue(Group::GENERAL).toString(), settings))
    , m_dewarpingConfig(std::make_shared<Model::DewarpingConfig>(QVariant::fromValue(Group::DEWARPING).toString(), settings))
    , m_videoInputConfig(std::make_shared<Model::VideoConfig>(QVariant::fromValue(Group::VIDEO_INPUT).toString(), settings))
    , m_videoOutputConfig(std::make_shared<Model::VideoConfig>(QVariant::fromValue(Group::VIDEO_OUTPUT).toString(), settings))
    , m_audioInputConfig(std::make_shared<Model::AudioConfig>(QVariant::fromValue(Group::AUDIO_INPUT).toString(), settings))
    , m_audioOutputConfig(std::make_shared<Model::AudioConfig>(QVariant::fromValue(Group::AUDIO_OUTPUT).toString(), settings))
    , m_streamConfig(std::make_shared<Model::StreamConfig>(QVariant::fromValue(Group::STREAM).toString(), settings))
    , m_transcriptionConfig(std::make_shared<Model::TranscriptionConfig>(QVariant::fromValue(Group::TRANSCRIPTION).toString(), settings))
{
    addSubConfig(m_appConfig);
    addSubConfig(m_dewarpingConfig);
    addSubConfig(m_videoInputConfig);
    addSubConfig(m_videoOutputConfig);
    addSubConfig(m_audioInputConfig);
    addSubConfig(m_audioOutputConfig);
    addSubConfig(m_streamConfig);
    addSubConfig(m_transcriptionConfig);

    // Do not override an existing file.
    // Maybe the user created a custom config file.
    if (!QFileInfo::exists(APP_CONFIG_FILE))
    {
        loadDefault();
    }
}

const AppConfig& Settings::appConfig() const
{
    return *m_appConfig;
}

AppConfig& Settings::appConfig()
{
    return *m_appConfig;
}

const DewarpingConfig& Settings::dewarpingConfig() const
{
    return *m_dewarpingConfig;
}

DewarpingConfig& Settings::dewarpingConfig()
{
    return *m_dewarpingConfig;
}

const VideoConfig& Settings::videoInputConfig() const
{
    return *m_videoInputConfig;
}

VideoConfig& Settings::videoInputConfig()
{
    return *m_videoInputConfig;
}

const VideoConfig& Settings::videoOutputConfig() const
{
    return *m_videoOutputConfig;
}

VideoConfig& Settings::videoOutputConfig()
{
    return *m_videoOutputConfig;
}

const AudioConfig& Settings::audioInputConfig() const
{
    return *m_audioInputConfig;
}

AudioConfig& Settings::audioInputConfig()
{
    return *m_audioInputConfig;
}

const AudioConfig& Settings::audioOutputConfig() const
{
    return *m_audioOutputConfig;
}

AudioConfig& Settings::audioOutputConfig()
{
    return *m_audioOutputConfig;
}

const StreamConfig& Settings::streamConfig() const
{
    return *m_streamConfig;
}

StreamConfig& Settings::streamConfig()
{
    return *m_streamConfig;
}

const TranscriptionConfig& Settings::transcriptionConfig() const
{
    return *m_transcriptionConfig;
}

TranscriptionConfig& Settings::transcriptionConfig()
{
    return *m_transcriptionConfig;
}

void Settings::loadDefault()
{
    m_appConfig->setValue(AppConfig::Key::OUTPUT_FOLDER, QDir::homePath());

    m_transcriptionConfig->setValue(TranscriptionConfig::Key::LANGUAGE, Transcription::Language::FR_CA);
    m_transcriptionConfig->setValue(TranscriptionConfig::Key::AUTOMATIC_TRANSCRIPTION, false);

    m_dewarpingConfig->setValue(DewarpingConfig::Key::IN_RADIUS, 400);
    m_dewarpingConfig->setValue(DewarpingConfig::Key::OUT_RADIUS, 1400);
    m_dewarpingConfig->setValue(DewarpingConfig::Key::ANGLE_SPAN, 90);
    m_dewarpingConfig->setValue(DewarpingConfig::Key::TOP_DISTORSION_FACTOR, 0.08);
    m_dewarpingConfig->setValue(DewarpingConfig::Key::BOTTOM_DISTORSION_FACTOR, 0);
    m_dewarpingConfig->setValue(DewarpingConfig::Key::FISH_EYE_ANGLE, 220);
    m_dewarpingConfig->update();

    m_videoInputConfig->setValue(VideoConfig::Key::FPS, 20);
    m_videoInputConfig->setValue(VideoConfig::Key::WIDTH, 2880);
    m_videoInputConfig->setValue(VideoConfig::Key::HEIGHT, 2160);
    m_videoInputConfig->setValue(VideoConfig::Key::DEVICE_NAME, "/dev/video0");
    m_videoInputConfig->setValue(VideoConfig::Key::IMAGE_FORMAT, ImageFormat::UYVY_FMT);
    m_videoInputConfig->update();

    m_videoOutputConfig->setValue(VideoConfig::Key::FPS, 20);
    m_videoOutputConfig->setValue(VideoConfig::Key::WIDTH, 800);
    m_videoOutputConfig->setValue(VideoConfig::Key::HEIGHT, 600);
    m_videoOutputConfig->setValue(VideoConfig::Key::DEVICE_NAME, "/dev/video0");
    m_videoOutputConfig->setValue(VideoConfig::Key::IMAGE_FORMAT, ImageFormat::UYVY_FMT);
    m_videoOutputConfig->update();

    m_audioInputConfig->setValue(AudioConfig::Key::DEVICE_NAME, "odas");
    m_audioInputConfig->setValue(AudioConfig::Key::CHANNELS, 4);
    m_audioInputConfig->setValue(AudioConfig::Key::RATE, 44100);
    m_audioInputConfig->setValue(AudioConfig::Key::FORMAT_BYTES, 2);
    m_audioInputConfig->setValue(AudioConfig::Key::IS_LITTLE_ENDIAN, true);
    m_audioInputConfig->setValue(AudioConfig::Key::PACKET_AUDIO_SIZE, 4096);
    m_audioInputConfig->setValue(AudioConfig::Key::PACKET_HEADER_SIZE, 0);
    m_audioInputConfig->update();

    m_audioOutputConfig->setValue(AudioConfig::Key::DEVICE_NAME, "");
    m_audioOutputConfig->setValue(AudioConfig::Key::CHANNELS, 4);
    m_audioOutputConfig->setValue(AudioConfig::Key::RATE, 44100);
    m_audioOutputConfig->setValue(AudioConfig::Key::FORMAT_BYTES, 2);
    m_audioOutputConfig->setValue(AudioConfig::Key::IS_LITTLE_ENDIAN, true);
    m_audioOutputConfig->setValue(AudioConfig::Key::PACKET_AUDIO_SIZE, 4096);
    m_audioOutputConfig->setValue(AudioConfig::Key::PACKET_HEADER_SIZE, 0);
    m_audioOutputConfig->update();

    m_streamConfig->setValue(StreamConfig::Key::DETECTION_DEWARPING_COUNT, 4);
    m_streamConfig->setValue(StreamConfig::Key::ASPECT_RATIO_WIDTH, 3);
    m_streamConfig->setValue(StreamConfig::Key::ASPECT_RATIO_HEIGHT, 4);
    m_streamConfig->setValue(StreamConfig::Key::MIN_ELEVATION, 0);
    m_streamConfig->setValue(StreamConfig::Key::MAX_ELEVATION, 90);
    m_streamConfig->update();
}

}    // namespace Model
