#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>

#include <QMainWindow>

#include "model/stream/i_stream.h"

class QStackedWidget;

namespace Model
{
class ISettings;
class IMediaPlayer;
}    // namespace Model

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
    MainWindow(Model::ISettings &settings, Model::IMediaPlayer &mediaPlayer, std::shared_ptr<Model::IStream> stream,
               QWidget *parent = nullptr);

   private:
    void addView(View::AbstractView *view);

    Ui::MainWindow *m_ui;
    View::SideBar *m_sideBar;
    QStackedWidget *m_views;
};

}    // namespace View

#endif    // MAINWINDOW_H
