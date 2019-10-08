#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFile>
#include <QObject>
#include <QStackedWidget>

#include "view/components/sidebar.h"
#include "view/views/abstract_view.h"
#include "view/views/conference_view.h"
#include "view/views/recording_view.h"
#include "view/views/media_player_view.h"
#include "view/views/transcription_view.h"
#include "view/views/settings_view.h"

#include "model/media_player.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , sideBar(new View::SideBar)
    , views(new QStackedWidget)
    , conferenceView(new View::ConferenceView)
    , recordingView(new View::RecordingView)
    , transcriptionView(new View::TranscriptionView)
    , settingsView(new View::SettingsView())
{
    ui->setupUi(this);

    auto mediaPlayer = new Model::MediaPlayer();
    mediaPlayerView = new View::MediaPlayerView(*mediaPlayer);

    //Model::ISettingsView *settingsView


    QFile File("view/stylesheets/globalStylesheet.qss");
    File.open(QFile::ReadOnly);
    this->setStyleSheet(QLatin1String(File.readAll()));

    addView(conferenceView);
    addView(recordingView);
    addView(mediaPlayerView);
    addView(transcriptionView);
    addView(settingsView);

    ui->mainLayout->addWidget(sideBar);
    ui->mainLayout->addWidget(views);

    sideBar->setCurrentRow(0);

    connect(sideBar, &View::SideBar::currentRowChanged,
            [=](const int& index){ views->setCurrentIndex(index); });
}

void MainWindow::addView(View::AbstractView *view)
{
    sideBar->add(view->getName());
    views->addWidget(view);
}
