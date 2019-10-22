#ifndef RECORDER_H
#define RECORDER_H

#include "i_recorder.h"

class QMediaRecorder;
class QCamera;
class QCameraInfo;
//class QAudioRecorder;

namespace Model
{

class Recorder : public IRecorder
{
public:
    explicit Recorder(QWidget *parent = nullptr);
    void start(std::string outputPath) override;
    void stop() override;

private:
    QCameraInfo getCameraInfo();
    QMediaRecorder *m_mediaRecorder;
    QCamera *m_camera;
};

} // Model

#endif // RECORDER_H
