#include "local_conference_view.h"
#include "ui_local_conference_view.h"
#include "model/recorder/recorder.h"
#include "model/settings/settings_constants.h"
#include "model/settings/i_settings.h"

#include <QCamera>
#include <QCameraInfo>
#include <QCameraViewfinder>
#include <QStateMachine>
#include <QState>
#include <QUrl>

namespace View
{

LocalConferenceView::LocalConferenceView(Model::ISettings& settings, QWidget *parent)
    : AbstractView("Local Conference", parent)
    , m_ui(new Ui::LocalConferenceView)
    , m_settings(settings)
    , m_camera(new QCamera(getCameraInfo(), this))
    , m_cameraViewfinder(new QCameraViewfinder(this))
    , m_recorder(new Model::Recorder(m_camera, this))
    , m_stateMachine(new QStateMachine)
    , m_stopped(new QState)
    , m_started(new QState)
{
    m_ui->setupUi(this);
    m_ui->virtualCameraLayout->addWidget(m_cameraViewfinder);

    m_camera->setCaptureMode(QCamera::CaptureVideo);
    m_camera->setViewfinder(m_cameraViewfinder);
    m_cameraViewfinder->show();

    m_stopped->assignProperty(m_ui->btnStartStopRecord, "text", "Start recording");
    m_started->assignProperty(m_ui->btnStartStopRecord, "text", "Stop recording");

    m_stopped->addTransition(m_ui->btnStartStopRecord, &QAbstractButton::clicked, m_started);
    m_started->addTransition(m_ui->btnStartStopRecord, &QAbstractButton::clicked, m_stopped);

    m_stateMachine->addState(m_stopped);
    m_stateMachine->addState(m_started);

    m_stateMachine->setInitialState(m_stopped);
    m_stateMachine->start();

    connect(m_started, &QState::entered, [=]{ m_recorder->start(getOutputPath()); });
    connect(m_stopped, &QState::entered, [=]{ m_recorder->stop(); });
}

LocalConferenceView::~LocalConferenceView()
{
    stopCamera();
}

void LocalConferenceView::showEvent(QShowEvent */*event*/)
{
    startCamera();
}

void LocalConferenceView::hideEvent(QHideEvent */*event*/)
{
    stopCamera();
}

QString LocalConferenceView::getOutputPath()
{
    return m_settings.get(Model::General::keyName(Model::General::Key::OUTPUT_FOLDER)).toString();
}

QCameraInfo LocalConferenceView::getCameraInfo()
{
    QCameraInfo defaultCameraInfo = QCameraInfo::defaultCamera();

    if(!Model::VIRTUAL_CAMERA_DEVICE.isEmpty())
    {
        QList<QCameraInfo> cameras = QCameraInfo::availableCameras();

        foreach (const QCameraInfo &cameraInfo, cameras)
        {
            if (cameraInfo.deviceName() == Model::VIRTUAL_CAMERA_DEVICE)
                return cameraInfo;
        }
    }

    return defaultCameraInfo;
}

void LocalConferenceView::startCamera()
{
    if(m_camera->state() != QCamera::State::ActiveState)
    {
        m_camera->start();
    }
}

void LocalConferenceView::stopCamera()
{
    if(m_camera->state() == QCamera::State::ActiveState)
    {
        m_camera->stop();
    }
}

} // View
