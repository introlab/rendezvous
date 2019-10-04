#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFile>
#include <QObject>
#include <QStackedWidget>

#include "view/components/sidebar.h"
#include "view/views/conference.h"
#include "view/views/recording.h"
#include "view/views/audio_processing.h"
#include "view/views/transcription.h"
#include "view/views/settings.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , views(new QStackedWidget)
    , sideBar(new View::SideBar)
    , conferenceView(new View::Conference)
    , recordingView(new View::Recording)
    , audioProcessingView(new View::AudioProcessing)
    , transcriptionView(new View::Transcription)
    , settingsView(new View::Settings)
{
    ui->setupUi(this);

    QFile File("view/stylesheets/globalStylesheet.qss");
    File.open(QFile::ReadOnly);
    qApp->setStyleSheet(QLatin1String(File.readAll()));

    addView("Conference", conferenceView);
    addView("Recording", recordingView);
    addView("Audio Processing", audioProcessingView);
    addView("Transcription", transcriptionView);
    addView("Settings", settingsView);

    sideBar->setCurrentRow(0);
    ui->mainLayout->addWidget(sideBar);
    ui->mainLayout->addWidget(views);

    connect(sideBar, &View::SideBar::currentRowChanged,
            [=]( const int& index) { views->setCurrentIndex(index); });
}

void MainWindow::addView(const QString &name, QWidget *view)
{
    sideBar->add(name);
    views->addWidget(view);
}


MainWindow::~MainWindow()
{
    delete ui;
}
