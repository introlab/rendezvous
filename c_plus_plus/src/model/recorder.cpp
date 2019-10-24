#include "recorder.h"

#include <QMediaRecorder>
#include <QCamera>
#include <QCameraInfo>
#include <QCameraViewfinder>
#include <QUrl>
#include <QVideoEncoderSettings>
#include <QProcess>

namespace Model
{

Recorder::Recorder(const QString cameraDevice, const QString audioDevice, QWidget *parent)
    : IRecorder(parent)
    , m_camera(new QCamera(getCameraInfo(cameraDevice), this))
    , m_process(new QProcess(this))
    , m_mediaRecorder(new QMediaRecorder(m_camera, this))
    , m_audioRecorder(new QAudioRecorder(this))
{
    m_camera->setCaptureMode(QCamera::CaptureMode::CaptureVideo);

    // Video
    QVideoEncoderSettings videoSettings;
    videoSettings.setCodec("video/x-h264");
    m_mediaRecorder->setVideoSettings(videoSettings);
    m_mediaRecorder->setMuted(true);
    m_mediaRecorder->setContainerFormat("avi");

    // Audio
    m_audioRecorder->setContainerFormat("audio/x-wav");
    m_audioRecorder->setAudioInput(getAudioInput(audioDevice));

    connect(m_audioRecorder, &QAudioRecorder::stateChanged,
            [=](QAudioRecorder::State state){ onAudioRecorderStateChanged(state); });
}

void Recorder::start(const QString outputPath)
{
    m_outputPath = outputPath;

    QUrl videoPath = QUrl::fromLocalFile(outputPath + "/video.avi");
    QUrl audioPath = QUrl::fromLocalFile(outputPath + "/audio.wav");

    m_mediaRecorder->setOutputLocation(videoPath);
    m_audioRecorder->setOutputLocation(audioPath);

    if(m_mediaRecorder->isAvailable() && m_camera->status() == QCamera::Status::ActiveStatus)
    {
        m_mediaRecorder->record();
        m_audioRecorder->record();
    }
}

void Recorder::stop()
{
    m_mediaRecorder->stop();
    m_audioRecorder->stop();
}

QCameraInfo Recorder::getCameraInfo(QString cameraDevice)
{
    QCameraInfo defaultCameraInfo = QCameraInfo::defaultCamera();

    if(cameraDevice != "")
    {
        QList<QCameraInfo> cameras = QCameraInfo::availableCameras();

        foreach (const QCameraInfo &cameraInfo, cameras)
        {
            if (cameraInfo.deviceName() == cameraDevice)
                return cameraInfo;
        }
    }

    return defaultCameraInfo;
}

QString Recorder::getAudioInput(const QString audioDevice)
{
    QString defaultInput = m_audioRecorder->defaultAudioInput();

    if(audioDevice != "")
    {
        QStringList inputs = m_audioRecorder->audioInputs();

        foreach (QString input, inputs)
        {
            if(input == audioDevice)
                return input;
        }
    }

    return defaultInput;
}

void Recorder::startCamera()
{
    if(m_camera->state() != QCamera::State::ActiveState)
    {
        m_camera->start();
    }
}

void Recorder::stopCamera()
{
    if(m_camera->state() == QCamera::State::ActiveState)
    {
        m_camera->stop();
    }
}

void Recorder::setCameraViewfinder(QCameraViewfinder *viewfinder)
{
    m_camera->setViewfinder(viewfinder);
}

void Recorder::mergeAudioVideo()
{
    QStringList arguments;
    arguments << "-loglevel" << "panic" << "-i" << m_outputPath + "/audio.wav" << "-i" << m_outputPath + "/video.avi"
              << "-vcodec" << "copy" << "-acodec" << "copy" << m_outputPath + "/media.avi" << "-y";

    int ret = m_process->execute("ffmpeg", arguments);

    if(ret == 0)
    {
        m_process->execute("rm " + m_outputPath + "/audio.wav");
        m_process->execute("rm " + m_outputPath + "/video.avi");
    }

}

void Recorder::onAudioRecorderStateChanged(QAudioRecorder::State state)
{
    if(state == QMediaRecorder::StoppedState)
    {
        QMediaRecorder::State mediaState = m_mediaRecorder->state();
        if( mediaState == QMediaRecorder::StoppedState)
        {
            mergeAudioVideo();
        }
        else
        {
            qWarning("Audio and video were not merged");
        }
    }
}

} // Model
