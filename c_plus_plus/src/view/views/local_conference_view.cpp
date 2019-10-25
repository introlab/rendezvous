#include "local_conference_view.h"
#include "ui_local_conference_view.h"
#include "model/settings/settings_constants.h"

#include <QCameraViewfinder>
#include <QUrl>

namespace View
{

LocalConferenceView::LocalConferenceView(Model::ISettings& settings, QWidget *parent)
    : AbstractView("Local Conference", parent)
    , m_ui(new Ui::LocalConferenceView)
    , m_recorder(new Model::Recorder(getCameraDevice(), this))
    , m_settings(&settings)
    , m_cameraViewfinder(new QCameraViewfinder(this))
{
    m_ui->setupUi(this);
    m_ui->virtualCameraLayout->addWidget(m_cameraViewfinder);

    m_recorder->setCameraViewfinder(m_cameraViewfinder);
    m_cameraViewfinder->show();

    connect(m_ui->btnStartStopRecord, &QAbstractButton::clicked, [=]{ changeRecordButtonState(); });
}

void LocalConferenceView::changeRecordButtonState()
{
    m_recordButtonState = !m_recordButtonState;

    if(m_recordButtonState)
    {
        m_ui->btnStartStopRecord->setText("Stop recording");
        m_recorder->start(getOutputPath());
    }
    else
    {
        m_ui->btnStartStopRecord->setText("Start recording");
        m_recorder->stop();
    }
}

void LocalConferenceView::showEvent(QShowEvent */*event*/)
{
    m_recorder->startCamera();
}

void LocalConferenceView::hideEvent(QHideEvent */*event*/)
{
    m_recorder->stopCamera();
}

QString LocalConferenceView::getCameraDevice()
{
    return Model::VIRTUAL_CAMERA_DEVICE;
}

QString LocalConferenceView::getOutputPath()
{
    return m_settings->get(Model::General::keyName(Model::General::Key::OUTPUT_FOLDER)).toString();
}

} // View
