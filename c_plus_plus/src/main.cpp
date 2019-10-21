#include "view/mainwindow.h"

#include <QApplication>

#include "model/settings/settings.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    Model::Settings settings;
    MainWindow w(settings);
    w.show();

    return QApplication::exec();
}
