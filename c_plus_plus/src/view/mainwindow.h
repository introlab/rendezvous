#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class QStackedWidget;

namespace Model { class ISettings;
                  class IMediaPlayer; }

namespace View { class SideBar;
                 class AbstractView; }

namespace Ui { class MainWindow; }

namespace View
{

class MainWindow : public QMainWindow
{
Q_OBJECT

public:
    MainWindow(Model::ISettings &settings, Model::IMediaPlayer &mediaPlayer, QWidget *parent = nullptr);

private:
    void addView(View::AbstractView *view);

    Ui::MainWindow *ui;
    View::SideBar *sideBar;
    QStackedWidget *views;
};

} // View

#endif // MAINWINDOW_H
