#ifndef RECORDER_H
#define RECORDER_H

#include "i_recorder.h"

class QMediaRecorder;
class QCamera;

namespace Model
{
class Recorder : public IRecorder
{
   public:
    explicit Recorder(QCamera *camera, QWidget *parent = nullptr);
    void start(const QString outputPath) override;
    void stop() override;

   public slots:
    void onStartRecording();
    void onStopRecording();

   private:
    QCamera *m_camera;
    QMediaRecorder *m_mediaRecorder;
};

}    // namespace Model

#endif    // RECORDER_H
