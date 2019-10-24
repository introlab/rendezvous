#ifndef LOCAL_CONFERENCE_VIEW_H
#define LOCAL_CONFERENCE_VIEW_H

#include "view/views/abstract_view.h"
#include "model/settings/i_settings.h"
#include "model/recorder.h"

class QCameraViewfinder;

namespace Ui { class LocalConferenceView; }

namespace View
{

class LocalConferenceView : public AbstractView
{
public:
    explicit LocalConferenceView(Model::ISettings& settings, QWidget *parent = nullptr);

public slots:
    void changeRecordButtonState();

protected:
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;

private:
    QString getCameraDevice();
    QString getAudioDevice();
    QString getOutputPath();

    Ui::LocalConferenceView *m_ui;
    Model::Recorder *m_recorder;
    Model::ISettings *m_settings;
    QCameraViewfinder *m_cameraViewfinder;
    bool m_recordButtonState = false;
};

} // View

#endif // LOCAL_CONFERENCE_VIEW_H