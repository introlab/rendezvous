#include "online_conference_view.h"
#include "ui_online_conference_view.h"

#include <stdexcept>

#include <QDesktopServices>
#include <QUrl>

namespace View
{
OnlineConferenceView::OnlineConferenceView(std::shared_ptr<Model::IStream> stream, QWidget *parent)
    : AbstractView("Online Conference", parent)
    , m_ui(new Ui::OnlineConferenceView)
    , m_stream(stream)
{
    if (m_stream == nullptr)
    {
        throw std::invalid_argument("Stream is null!");
    }

    m_ui->setupUi(this);

    connect(m_ui->websiteButton, &QAbstractButton::clicked,
            [] { QDesktopServices::openUrl(QUrl("https://rendezvous-meet.com/")); });

    connect(m_stream.get(), &Model::IStream::stateChanged, [=](const Model::IStream::State& state){ onStreamStateChanged(state); });
    connect(m_ui->startButton, &QAbstractButton::clicked, [=] { onStartButtonClicked(); });
}

OnlineConferenceView::~OnlineConferenceView()
{
    m_stream->stop();
}

void OnlineConferenceView::onStartButtonClicked()
{   
    m_ui->startButton->setDisabled(true);
    switch (m_stream->state())
    {
        case Model::IStream::Started:
            m_stream->stop();
            break;
        case Model::IStream::Stopped:
            m_stream->start();
            break;
    }
}

void OnlineConferenceView::onStreamStateChanged(const Model::IStream::State& state)
{
    switch (state)
    {
        case Model::IStream::Started:
            m_ui->startButton->setText("Stop virtual devices");
            break;
        case Model::IStream::Stopped:
            m_ui->startButton->setText("Start virtual devices");
            break;
    }
    m_ui->startButton->setDisabled(false);
}

}    // namespace View
