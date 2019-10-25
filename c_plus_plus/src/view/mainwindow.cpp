#include "mainwindow.h"

#include "ui_mainwindow.h"

#include "components/sidebar.h"
#include "model/media_player/i_media_player.h"
#include "views/abstract_view.h"
#include "views/local_conference_view.h"
#include "views/media_player_view.h"
#include "views/online_conference_view.h"
#include "views/settings_view.h"

#include <QStackedWidget>

namespace View
{

MainWindow::MainWindow(Model::ISettings &settings, Model::IMediaPlayer &mediaPlayer, QWidget *parent)
    : QMainWindow(parent)
    , m_ui(new Ui::MainWindow)
    , m_sideBar(new View::SideBar)
    , m_views(new QStackedWidget)
{
    m_ui->setupUi(this);
    m_ui->mainLayout->addWidget(m_sideBar);
    m_ui->mainLayout->addWidget(m_views);

    View::AbstractView *onlineConferenceView = new View::OnlineConferenceView();
    View::AbstractView *localConferenceView = new View::LocalConferenceView();
    View::AbstractView *mediaPlayerView = new View::MediaPlayerView(mediaPlayer);
    View::AbstractView *settingsView = new View::SettingsView(settings);

    addView(onlineConferenceView);
    addView(localConferenceView);
    addView(mediaPlayerView);
    addView(settingsView);

    m_sideBar->setCurrentRow(0);

    connect(m_sideBar, &View::SideBar::currentRowChanged, [=](const int& index){ m_views->setCurrentIndex(index); });
}

void MainWindow::addView(View::AbstractView *view)
{
    m_sideBar->add(view->getName());
    m_views->addWidget(view);
}

} // View
