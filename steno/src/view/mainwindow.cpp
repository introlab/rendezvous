#include "mainwindow.h"

#include "ui_mainwindow.h"

#include "components/side_bar.h"
#include "components/top_bar.h"
#include "model/media_player/i_media_player.h"
#include "views/abstract_view.h"
#include "views/local_conference_view.h"
#include "views/media_player_view.h"
#include "views/online_conference_view.h"
#include "views/settings_view.h"

#include <QStackedWidget>

namespace View
{
MainWindow::MainWindow(std::shared_ptr<Model::Config> config, std::shared_ptr<Model::IMediaPlayer> mediaPlayer,
                       std::shared_ptr<Model::IStream> stream, std::shared_ptr<Model::IRecorder> recorder,
                       QWidget *parent)
    : QMainWindow(parent)
    , m_ui(new Ui::MainWindow)
    , m_sideBar(new View::SideBar)
    , m_topBar(new View::TopBar)
    , m_views(new QStackedWidget)
{
    m_ui->setupUi(this);

    m_ui->leftLayout->addWidget(m_sideBar);
    m_ui->rightLayout->addWidget(m_topBar);
    m_ui->rightLayout->addWidget(m_views);

    View::AbstractView *onlineConferenceView = new View::OnlineConferenceView(stream);
    View::AbstractView *localConferenceView = new View::LocalConferenceView(stream, recorder);
    View::AbstractView *mediaPlayerView = new View::MediaPlayerView(mediaPlayer);
    View::AbstractView *settingsView = new View::SettingsView(config);

    addView(onlineConferenceView, QIcon(":/icons/meeting.svg"));
    addView(localConferenceView, QIcon(":/icons/meeting.svg"));
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

}    // namespace View
