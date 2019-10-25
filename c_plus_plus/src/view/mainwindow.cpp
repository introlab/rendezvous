#include "mainwindow.h"

#include "ui_mainwindow.h"

#include "components/sidebar.h"
#include "model/media_player/i_media_player.h"
#include "views/abstract_view.h"
#include "views/local_conference_view.h"
#include "views/media_player_view.h"
#include "views/online_conference_view.h"
#include "views/settings_view.h"

#include <QDir>
#include <QFile>
#include <QObject>
#include <QStackedWidget>

namespace View
{

MainWindow::MainWindow(Model::ISettings &settings, Model::IMediaPlayer &mediaPlayer, QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , sideBar(new View::SideBar)
    , views(new QStackedWidget)
{

    ui->setupUi(this);
    ui->mainLayout->addWidget(sideBar);
    ui->mainLayout->addWidget(views);

    QFile File(QDir::currentPath() + "/../src/view/stylesheets/globalStylesheet.qss");
    File.open(QFile::ReadOnly);
    this->setStyleSheet(QLatin1String(File.readAll()));

    View::AbstractView *onlineConferenceView = new View::OnlineConferenceView();
    View::AbstractView *localConferenceView = new View::LocalConferenceView();
    View::AbstractView *mediaPlayerView = new View::MediaPlayerView(mediaPlayer);
    View::AbstractView *settingsView = new View::SettingsView(settings);

    addView(onlineConferenceView);
    addView(localConferenceView);
    addView(mediaPlayerView);
    addView(settingsView);

    sideBar->setCurrentRow(0);

    connect(sideBar, &View::SideBar::currentRowChanged, [=](const int& index){ views->setCurrentIndex(index); });
}

void MainWindow::addView(View::AbstractView *view)
{
    sideBar->add(view->getName());
    views->addWidget(view);
}

} // View
