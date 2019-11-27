#include "model/config/config.h"
#include "model/media/media.h"
#include "model/media_player/media_player.h"
#include "model/stream/stream.h"
#include "model/stream/video/output/default_virtual_camera_output.h"
#include "model/transcription/transcription.h"
#include "view/mainwindow.h"

#include <memory>

#include <QApplication>
#include <QSettings>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    std::shared_ptr<Model::IMediaPlayer> mediaPlayer = std::make_shared<Model::MediaPlayer>();

    const QString configFile = QCoreApplication::applicationDirPath() + "/../steno.conf";

    std::shared_ptr<QSettings> qSettings = std::make_shared<QSettings>(configFile, QSettings::IniFormat);

    std::shared_ptr<Model::Config> config = std::make_shared<Model::Config>(qSettings, configFile);

    std::shared_ptr<Model::IStream> stream = std::make_shared<Model::Stream>(config);

    std::shared_ptr<Model::Media> media = std::make_shared<Model::Media>(config, stream);

    std::shared_ptr<Model::Transcription> transcription = std::make_shared<Model::Transcription>(config);

    View::MainWindow w(config, mediaPlayer, stream, media, transcription);
    w.show();

    Model::DefaultVirtualCameraOutput::writeDefaultImage(
        config->videoOutputConfig()->value(Model::VideoConfig::DEVICE_NAME).toString());

    return QApplication::exec();
}
