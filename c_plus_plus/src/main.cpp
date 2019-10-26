#include "model/media_player/media_player.h"
#include "model/settings/settings.h"
#include "view/mainwindow.h"

#include <QApplication>
#include <QFile>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QFile file(   ":/stylesheets/globalStylesheet.qss");
    file.open(QFile::ReadOnly);
    a.setStyleSheet(QLatin1String(file.readAll()));

    Model::Settings settings;
    Model::MediaPlayer mediaPlayer;

    View::MainWindow w(settings, mediaPlayer);
    w.show();

    return QApplication::exec();
}
