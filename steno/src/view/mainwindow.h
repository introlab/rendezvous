#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>

#include <QMainWindow>

#include "model/media_player/i_media_player.h"
#include "model/recorder/i_recorder.h"
#include "model/settings/settings.h"
#include "model/stream/i_stream.h"

class QStackedWidget;

namespace View
{
class SideBar;
class AbstractView;
}    // namespace View

namespace Ui
{
class MainWindow;
}

namespace View
{
class MainWindow : public QMainWindow
{
    Q_OBJECT

   public:
    MainWindow(std::shared_ptr<Model::Settings> settings, std::shared_ptr<Model::IMediaPlayer> mediaPlayer,
               std::shared_ptr<Model::IStream> stream, std::shared_ptr<Model::IRecorder> recorder,
               QWidget *parent = nullptr);

   private:
    void addView(View::AbstractView *view);

    Ui::MainWindow *m_ui;
    View::SideBar *m_sideBar;
    QStackedWidget *m_views;
};

}    // namespace View

#endif    // MAINWINDOW_H
