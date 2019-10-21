#ifndef RECORDING_VIEW_H
#define RECORDING_VIEW_H

#include <QCamera>
#include <QCameraInfo>
#include <QCameraViewfinder>
#include <QListWidgetItem>

#include "view/views/abstract_view.h"

namespace Ui { class RecordingView; }

namespace View
{

class RecordingView : public AbstractView
{
public:
    explicit RecordingView(QWidget *parent = nullptr);

protected:
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;

private:
    QCameraInfo getCameraInfo();
    void changeRecordButtonState();

    Ui::RecordingView *mUi;
    QCamera *mCamera;
    QCameraViewfinder *mCameraViewfinder;
    bool mRecordButtonState = false;
};

} // View

#endif // RECORDING_VIEW_H
