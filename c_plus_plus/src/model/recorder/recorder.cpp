#include "recorder.h"

#include <QCamera>
#include <QMediaRecorder>
#include <QUrl>

namespace Model
{
Recorder::Recorder(QCamera *camera, QWidget *parent)
    : IRecorder(parent)
    , m_camera(camera)
    , m_mediaRecorder(new QMediaRecorder(m_camera, this))
{
    QVideoEncoderSettings videoSettings;
    videoSettings.setQuality(QMultimedia::VeryHighQuality);
    videoSettings.setCodec("video/x-vp8");
    m_mediaRecorder->setVideoSettings(videoSettings);

    QAudioEncoderSettings audioSettings;
    audioSettings.setQuality(QMultimedia::VeryHighQuality);
    audioSettings.setCodec("audio/x-vorbis");
    m_mediaRecorder->setAudioSettings(audioSettings);

    m_mediaRecorder->setContainerFormat("video/webm");
}

void Recorder::start(const QString outputPath)
{
    if (!outputPath.isEmpty())
    {
        m_mediaRecorder->setOutputLocation(QUrl::fromLocalFile(outputPath + "/media"));
    }

    if (m_mediaRecorder->isAvailable() && m_camera->status() == QCamera::Status::ActiveStatus)
    {
        m_mediaRecorder->record();
    }
}

void Recorder::stop() { m_mediaRecorder->stop(); }
}    // namespace Model
