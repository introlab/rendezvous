#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class QStackedWidget;

namespace Ui { class MainWindow; }

namespace View { class SideBar;
                 class AbstractView; }

class MainWindow : public QMainWindow
{
Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);

private:
    void addView(View::AbstractView *view);

    Ui::MainWindow *ui;
    View::SideBar *sideBar;
    QStackedWidget *views;
    View::AbstractView *conferenceView;
    View::AbstractView *recordingView;
    View::AbstractView *mediaPlayerView;
    View::AbstractView *transcriptionView;
    View::AbstractView *settingsView;
};

#endif // MAINWINDOW_H
