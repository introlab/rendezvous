#ifndef RECORDER_H
#define RECORDER_H

#include "i_recorder.h"

#include <QAudioRecorder>

class QMediaRecorder;
class QCamera;
class QCameraInfo;
class QCameraViewfinder;
class QProcess;

namespace Model
{

class Recorder : public IRecorder
{
public:
    explicit Recorder(const QString cameraDevice, const QString audioDevice, QWidget *parent = nullptr);
    void start(const QString outputPath) override;
    void stop() override;

    void startCamera();
    void stopCamera();
    void setCameraViewfinder(QCameraViewfinder *viewfinder);

public slots:
    void onStartRecording();
    void onStopRecording();
    void onAudioRecorderStateChanged(QAudioRecorder::State state);

private:
    void mergeAudioVideo();
    QString getAudioInput(const QString audioDevice);
    QCameraInfo getCameraInfo(const QString cameraDevice);

    QCamera *m_camera;
    QProcess *m_process;
    QString m_outputPath;
    QMediaRecorder *m_mediaRecorder;
    QAudioRecorder *m_audioRecorder;
};

} // Model

#endif // RECORDER_H
