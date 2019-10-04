#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFile>
#include <QObject>
#include <QStackedWidget>

#include "view/components/sidebar.h"
#include "view/views/abstract_view.h"
#include "view/views/conference.h"
#include "view/views/recording.h"
#include "view/views/playback.h"
#include "view/views/transcription.h"
#include "view/views/settings.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , sideBar(new View::SideBar)
    , views(new QStackedWidget)
    , conferenceView(new View::Conference)
    , recordingView(new View::Recording)
    , playbackView(new View::Playback)
    , transcriptionView(new View::Transcription)
    , settingsView(new View::Settings)
{
    ui->setupUi(this);

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
            [=](const int& index) { views->setCurrentIndex(index); });
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
