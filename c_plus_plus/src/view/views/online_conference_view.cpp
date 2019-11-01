#include "online_conference_view.h"
#include "model/stream/i_stream.h"
#include "ui_online_conference_view.h"

#include <stdexcept>

#include <QDesktopServices>
#include <QSignalBlocker>
#include <QState>
#include <QStateMachine>
#include <QUrl>

namespace View
{
OnlineConferenceView::OnlineConferenceView(std::shared_ptr<Model::IStream> stream, QWidget* parent)
    : AbstractView("Online Conference", parent)
    , m_ui(new Ui::OnlineConferenceView)
    , m_stateMachine(new QStateMachine)
    , m_stopped(new QState)
    , m_started(new QState)
    , m_stream(stream)
{
    if (m_stream == nullptr)
    {
        throw std::invalid_argument("Stream is null!");
    }

    m_ui->setupUi(this);

    m_stopped->assignProperty(m_ui->startButton, "text", "Start virtual devices");
    m_started->assignProperty(m_ui->startButton, "text", "Stop virtual devices");

    m_stopped->addTransition(m_ui->startButton, &QAbstractButton::clicked, m_started);
    m_started->addTransition(m_ui->startButton, &QAbstractButton::clicked, m_stopped);
    m_started->addTransition(this, &OnlineConferenceView::streamCrashed, m_stopped);

    m_stateMachine->addState(m_stopped);
    m_stateMachine->addState(m_started);

    m_stateMachine->setInitialState(m_stopped);
    m_stateMachine->start();

    connect(m_ui->websiteButton, &QAbstractButton::clicked,
            [] { QDesktopServices::openUrl(QUrl("https://rendezvous-meet.com/")); });
    connect(m_stopped, &QState::entered, [=] { onStoppedStateEntered(); });
    connect(m_started, &QState::entered, [=] { onStartedStateEntered(); });

    connect(&*m_stream, &Model::IStream::statusChanged, this, [=] { onStreamStatusChanged(); });
}

OnlineConferenceView::~OnlineConferenceView()
{
    m_stream->stop();
}

void OnlineConferenceView::onStoppedStateEntered()
{
    m_currentState = m_stopped;
    m_ui->startButton->setDisabled(true);
    QApplication::processEvents();
    // We use a signal blocker to avoid queued signals from clicks on the startButton when the UI is disabled
    // The signals are reenable when the blocker is out of scope.
    QSignalBlocker blocker(m_ui->startButton);
    m_stream->stop();
}

void OnlineConferenceView::onStartedStateEntered()
{
    m_currentState = m_started;
    m_ui->startButton->setDisabled(true);
    QApplication::processEvents();
    m_stream->start();
}

void OnlineConferenceView::onStreamStatusChanged()
{
    const Model::StreamStatus status = m_stream->getStatus();
    switch (status)
    {
        case Model::StreamStatus::STOPPED:
            // there is a crash
            if (m_currentState == m_started)
            {
                emit streamCrashed(QPrivateSignal());
            }
        // intentionnal fall-through because both STOPPED and RUNNING need to reactivate the start button.
        case Model::StreamStatus::RUNNING:
            m_ui->startButton->setDisabled(false);
            QApplication::processEvents();
            break;
        case Model::StreamStatus::STOPPING:
            m_ui->startButton->setDisabled(true);
            QApplication::processEvents();
            break;
        default:
            break;
    }
}

}    // namespace View
