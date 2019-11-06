#include "model/media_player/media_player.h"
#include "model/recorder/recorder.h"
#include "model/config/config.h"
#include "model/app_constants.h"
#include "model/stream/stream.h"
#include "model/stream/video/output/default_virtual_camera_output.h"
#include "view/mainwindow.h"

#include <memory>

#include <QApplication>
#include <QFile>
#include <QSettings>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QFile file(":/stylesheets/globalStylesheet.qss");
    file.open(QFile::ReadOnly);
    a.setStyleSheet(QLatin1String(file.readAll()));

    std::shared_ptr<Model::IMediaPlayer> mediaPlayer = std::make_shared<Model::MediaPlayer>();

    std::shared_ptr<QSettings> qSettings = std::make_shared<QSettings>(Model::APP_CONFIG_FILE, QSettings::IniFormat);

    std::shared_ptr<Model::Config> settings = std::make_shared<Model::Config>(qSettings);

    std::shared_ptr<Model::IStream> stream = std::make_shared<Model::Stream>(
        settings->videoInputConfig(), settings->videoOutputConfig(), settings->audioInputConfig(),
        settings->audioOutputConfig(), settings->dewarpingConfig(), settings->streamConfig());

    std::shared_ptr<Model::IRecorder> recorder = std::make_shared<Model::Recorder>(settings);

    View::MainWindow w(settings, mediaPlayer, stream, recorder);
    w.show();

    Model::DefaultVirtualCameraOutput::writeDefaultImage();

    return QApplication::exec();
}
