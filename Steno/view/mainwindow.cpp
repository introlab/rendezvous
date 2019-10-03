#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFile>
#include <QObject>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , sideBar(parent)
{
    ui->setupUi(this);

    QFile File("stylesheet.qss");
    File.open(QFile::ReadOnly);
    qApp->setStyleSheet(QLatin1String(File.readAll()));

    sideBar.add("Conference");
    sideBar.add("Recording");
    sideBar.add("Audio Processing");
    sideBar.add("Transcription");
    sideBar.add("Web View");
    sideBar.add("Settings");
    sideBar.setCurrentRow(0);
    ui->mainLayout->addWidget(&sideBar);

    ui->mainLayout->addWidget(&views);

    /*
    connect(
        this, &SideBar::currentRowChanged,
        [=]( const int& index) { this->views.setCurrentIndex(index); }
    );*/

    //connect(&sideBar, &SideBar::currentRowChanged,
    //        &views, &MainWindow::onSideBarCurrentRowChanged);
}

MainWindow::~MainWindow()
{
    delete ui;
}


