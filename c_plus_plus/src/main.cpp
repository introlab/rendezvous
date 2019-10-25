#include "model/media_player/media_player.h"
#include "model/settings/settings.h"
#include "view/mainwindow.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    Model::Settings settings;
    Model::MediaPlayer mediaPlayer;

    View::MainWindow w(settings, mediaPlayer);
    w.show();

    return QApplication::exec();
}
