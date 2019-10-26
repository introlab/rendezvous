#ifndef RECORDING_VIEW_H
#define RECORDING_VIEW_H

#include "view/views/abstract_view.h"

class QCamera;
class QCameraInfo;
class QCameraViewfinder;
class QListWidgetItem;

namespace Ui
{
class LocalConferenceView;
}

namespace View
{
class LocalConferenceView : public AbstractView
{
   public:
    explicit LocalConferenceView(QWidget *parent = nullptr);

   public slots:
    void changeRecordButtonState();

   protected:
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;

   private:
    QCameraInfo getCameraInfo();

    Ui::LocalConferenceView *m_ui;
    QCamera *m_camera;
    QCameraViewfinder *m_cameraViewfinder;
    bool m_recordButtonState = false;
};

}    // View

#endif    // RECORDING_VIEW_H
