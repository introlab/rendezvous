#include "mainwindow.h"

#include "ui_mainwindow.h"

#include "components/side_bar.h"
#include "components/top_bar.h"
#include "model/media_player/i_media_player.h"
#include "model/transcription/transcription.h"
#include "views/abstract_view.h"
#include "views/conference_view.h"
#include "views/media_player_view.h"
#include "views/settings_view.h"

#include <QStackedWidget>

namespace View
{
MainWindow::MainWindow(std::shared_ptr<Model::Config> config, std::shared_ptr<Model::IMediaPlayer> mediaPlayer,
                       std::shared_ptr<Model::IStream> stream, std::shared_ptr<Model::Media> media,
                       std::shared_ptr<Model::Transcription> transcription, QWidget *parent)
    : QMainWindow(parent)
    , m_ui(new Ui::MainWindow)
    , m_sideBar(new View::SideBar(this))
    , m_topBar(new View::TopBar(stream, media, transcription, config, this))
    , m_views(new QStackedWidget(this))
{
    m_ui->setupUi(this);

    m_ui->leftLayout->addWidget(m_sideBar);
    m_ui->rightLayout->addWidget(m_topBar);
    m_ui->rightLayout->addWidget(m_views);

    View::AbstractView *conferenceView = new View::ConferenceView(media, this);
    View::AbstractView *mediaPlayerView = new View::MediaPlayerView(mediaPlayer, this);
    View::AbstractView *settingsView = new View::SettingsView(config, this);

    addView(conferenceView, QIcon(":/icons/meeting.svg"));
    addView(mediaPlayerView, QIcon(":/icons/player.svg"));
    addView(settingsView, QIcon(":/icons/settings.svg"));

    m_sideBar->setCurrentRow(0);

    setWindowIcon(QIcon(":/icons/app_icon.svg"));

    connect(m_sideBar, &View::SideBar::currentRowChanged, [=](const int &index) { m_views->setCurrentIndex(index); });
}

void MainWindow::addView(View::AbstractView *view, const QIcon &icon)
{
    m_sideBar->add(view->getName(), icon);
    m_views->addWidget(view);
}

void MainWindow::closeEvent(QCloseEvent * /*event*/)
{
    qDebug() << "closing application...";
    m_topBar->stopThreads();
}

}    // namespace View
