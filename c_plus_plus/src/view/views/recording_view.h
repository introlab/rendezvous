#ifndef RECORDING_VIEW_H
#define RECORDING_VIEW_H

#include "view/views/abstract_view.h"

class QCamera;
class QCameraInfo;
class QCameraViewfinder;
class QListWidgetItem;

namespace Ui { class RecordingView; }

namespace View
{

class RecordingView : public AbstractView
{
public:
    explicit RecordingView(QWidget *parent = nullptr);

public slots:
    void changeRecordButtonState();

protected:
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;

private:
    QCameraInfo getCameraInfo();

    Ui::RecordingView *m_ui;
    QCamera *m_camera;
    QCameraViewfinder *m_cameraViewfinder;
    bool m_recordButtonState = false;
};

} // View

#endif // RECORDING_VIEW_H
