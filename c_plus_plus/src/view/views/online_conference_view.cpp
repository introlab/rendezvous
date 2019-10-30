#include "online_conference_view.h"
#include "model/stream/i_stream.h"
#include "ui_online_conference_view.h"

#include <stdexcept>

#include <QDesktopServices>
#include <QState>
#include <QStateMachine>
#include <QUrl>

namespace View
{
OnlineConferenceView::OnlineConferenceView(std::shared_ptr<Model::IStream> stream, QWidget *parent)
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

    m_stateMachine->addState(m_stopped);
    m_stateMachine->addState(m_started);

    m_stateMachine->setInitialState(m_stopped);
    m_stateMachine->start();

    connect(m_ui->websiteButton, &QAbstractButton::clicked,
            [] { QDesktopServices::openUrl(QUrl("https://rendezvous-meet.com/")); });
    connect(m_stopped, &QState::entered, [=] { onStoppedStateEntered(); });
    connect(m_started, &QState::entered, [=] { onStartedStateEntered(); });
}

OnlineConferenceView::~OnlineConferenceView()
{
    m_stream->stop();
}

void OnlineConferenceView::onStoppedStateEntered()
{
    m_stream->stop();
}

void OnlineConferenceView::onStartedStateEntered()
{
    m_stream->start();
}

}    // namespace View
