#include "model/media_player/media_player.h"
#include "model/recorder/recorder.h"
#include "model/config/config.h"
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

    const QString configFile = QCoreApplication::applicationDirPath() + "/../steno.conf";

    std::shared_ptr<QSettings> qSettings = std::make_shared<QSettings>(configFile, QSettings::IniFormat);

    std::shared_ptr<Model::Config> config = std::make_shared<Model::Config>(qSettings, configFile);

    std::shared_ptr<Model::IStream> stream = std::make_shared<Model::Stream>(
        config->videoInputConfig(), config->videoOutputConfig(), config->audioInputConfig(),
        config->audioOutputConfig(), config->dewarpingConfig(), config->streamConfig(), config->appConfig());

    std::shared_ptr<Model::IRecorder> recorder = std::make_shared<Model::Recorder>(config);

    View::MainWindow w(config, mediaPlayer, stream, recorder);
    w.show();

    Model::DefaultVirtualCameraOutput::writeDefaultImage(config->videoOutputConfig().value(Model::VideoConfig::DEVICE_NAME).toString());

    return QApplication::exec();
}
