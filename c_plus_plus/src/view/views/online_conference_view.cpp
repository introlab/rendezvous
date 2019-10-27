#include "online_conference_view.h"
#include "model/stream/audio/odas/odas_client.h"
#include "ui_online_conference_view.h"

#include <QDesktopServices>
#include <QState>
#include <QStateMachine>
#include <QUrl>

namespace View
{
OnlineConferenceView::OnlineConferenceView(QWidget *parent)
    : AbstractView("Online Conference", parent)
    , m_ui(new Ui::OnlineConferenceView)
    , m_stateMachine(new QStateMachine)
    , m_stopped(new QState)
    , m_started(new QState)
    , m_odaslive(new Model::OdasClient)
{
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

void OnlineConferenceView::onStoppedStateEntered()
{
    // TODO stop virtual devices
    m_odaslive->stop();
}

void OnlineConferenceView::onStartedStateEntered()
{
    // TODO start virtual devices
    m_odaslive->start();
}

}    // namespace View
