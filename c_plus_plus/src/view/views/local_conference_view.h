#ifndef RECORDING_VIEW_H
#define RECORDING_VIEW_H

#include "view/views/abstract_view.h"

namespace Model
{
    class IRecorder;
    class ISettings;
}

class QCamera;
class QCameraInfo;
class QCameraViewfinder;
class QStateMachine;
class QState;

namespace Ui
{
class LocalConferenceView;
}

namespace View
{
class LocalConferenceView : public AbstractView
{
public:
    explicit LocalConferenceView(Model::ISettings& settings, QWidget *parent = nullptr);
    ~LocalConferenceView() override;

protected:
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;

private:
    QString getCameraDevice();
    QString getOutputPath();
    QCameraInfo getCameraInfo();
    void startCamera();
    void stopCamera();

    Ui::LocalConferenceView *m_ui;
    Model::ISettings& m_settings;
    QCamera *m_camera;
    QCameraViewfinder *m_cameraViewfinder;
    Model::IRecorder *m_recorder;
    QStateMachine *m_stateMachine;
    QState *m_stopped;
    QState *m_started;
};

}    // View

#endif    // RECORDING_VIEW_H
