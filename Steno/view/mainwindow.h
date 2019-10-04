#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class QStackedWidget;

namespace Ui { class MainWindow; }

namespace View {
                 class SideBar;
                 class Conference;
                 class Recording;
                 class AudioProcessing;
                 class Transcription;
                 class Settings;}

class MainWindow : public QMainWindow
{
    Q_OBJECT

    public:
        MainWindow(QWidget *parent = nullptr);
        virtual ~MainWindow();

    private:
        void addView(const QString &name, QWidget *view);

        Ui::MainWindow *ui;
        QStackedWidget *views;
        View::SideBar *sideBar;

        View::Conference *conferenceView;
        View::Recording *recordingView;
        View::AudioProcessing *audioProcessingView;
        View::Transcription *transcriptionView;
        View::Settings *settingsView;
};

#endif // MAINWINDOW_H
