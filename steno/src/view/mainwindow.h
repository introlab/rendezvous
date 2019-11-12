#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>

#include <QMainWindow>

#include "model/config/config.h"
#include "model/media_player/i_media_player.h"
#include "model/recorder/i_recorder.h"
#include "model/stream/i_stream.h"

class QStackedWidget;

namespace View
{
class SideBar;
class TopBar;
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
    MainWindow(std::shared_ptr<Model::Config> config, std::shared_ptr<Model::IMediaPlayer> mediaPlayer,
               std::shared_ptr<Model::IStream> stream, std::shared_ptr<Model::IRecorder> recorder,
               QWidget *parent = nullptr);

   private:
    void addView(View::AbstractView *view, const QIcon &icon);

    Ui::MainWindow *m_ui;
    View::SideBar *m_sideBar;
    View::TopBar *m_topBar;
    QStackedWidget *m_views;
};

}    // namespace View

#endif    // MAINWINDOW_H
