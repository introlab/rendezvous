#ifndef RECORDER_H
#define RECORDER_H

#include "i_recorder.h"
#include "model/config/config.h"

#include <memory>

#include <QCamera>
#include <QCameraInfo>
#include <QMediaRecorder>

namespace Model
{
class Recorder : public IRecorder
{
   public:
    explicit Recorder(std::shared_ptr<Model::Config> config, QWidget *parent = nullptr);
    ~Recorder() override;
    void start() override;
    void stop() override;
    void setCameraViewFinder(std::shared_ptr<QCameraViewfinder> cameraViewFinder) override;
    IRecorder::State state() const override
    {
        return m_state;
    }

   private slots:
    void onCameraStatusChanged(QCamera::Status status);

   private:
    void updateState(const IRecorder::State &state);
    QCameraInfo cameraInfo();
    void startCamera();
    void stopCamera();

    std::shared_ptr<Config> m_config;
    IRecorder::State m_state;
    QCamera m_camera;
    QMediaRecorder m_mediaRecorder;
};

}    // namespace Model

#endif    // RECORDER_H
