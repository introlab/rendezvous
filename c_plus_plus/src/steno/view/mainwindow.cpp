#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFile>
#include <QObject>
#include <QStackedWidget>

#include "view/components/sidebar.h"
#include "view/views/abstract_view.h"
#include "view/views/conference_view.h"
#include "view/views/recording_view.h"
#include "view/views/playback_view.h"
#include "view/views/transcription_view.h"
#include "view/views/settings_view.h"

#include "model/video_player.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , sideBar(new View::SideBar)
    , views(new QStackedWidget)
    , conferenceView(new View::ConferenceView)
    , recordingView(new View::RecordingView)
    , transcriptionView(new View::TranscriptionView)
    , settingsView(new View::SettingsView)
{
    ui->setupUi(this);

    Model::VideoPlayer *videoPlayer = new Model::VideoPlayer();
    playbackView = new View::PlaybackView(*videoPlayer);

    QFile File("view/stylesheets/globalStylesheet.qss");
    File.open(QFile::ReadOnly);
    qApp->setStyleSheet(QLatin1String(File.readAll()));

    addView(conferenceView);
    addView(recordingView);
    addView(playbackView);
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

MainWindow::~MainWindow()
{
    delete ui;
}
