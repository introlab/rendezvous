#ifndef RECORDER_H
#define RECORDER_H

#include "i_recorder.h"

class QMediaRecorder;
class QAudioRecorder;
class QCamera;
class QCameraInfo;
class QCameraViewfinder;

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

private:
    void setAudioInput(const QString audioDevice);
    QCameraInfo getCameraInfo(const QString cameraDevice);

    QCamera *m_camera;
    QMediaRecorder *m_mediaRecorder;
    QAudioRecorder *m_audioRecorder;
};

} // Model

#endif // RECORDER_H
