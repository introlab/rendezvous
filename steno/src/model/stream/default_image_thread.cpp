#include "default_image_thread.h"

#include "model/stream/video/input/image_file_reader.h"
#include "model/stream/video/output/virtual_camera_output.h"

#include <QFile>

namespace Model
{
DefaultImageThread::DefaultImageThread(std::shared_ptr<IVideoOutput> videoOutput,
                                       std::shared_ptr<VideoConfig> videoConfig)
    : m_videoOutput(videoOutput)
    , m_videoConfig(videoConfig)
{
}

void DefaultImageThread::run()
{
    qInfo() << "DefaulImageThread started";

    QFile file(":/defaultImage.jpg");
    if (!file.exists())
    {
        qCritical() << file.symLinkTarget() << "does not exists.";
        return;
    }

    ImageFormat imgFormat = static_cast<ImageFormat>(m_videoConfig->value(VideoConfig::IMAGE_FORMAT).toInt());
    ImageFileReader imageFileReader(file.symLinkTarget().toStdString(), imgFormat);
    imageFileReader.open();
    const Image& image = imageFileReader.readImage();
    imageFileReader.close();

    while (!isAbortRequested())
    {
        m_videoOutput->writeImage(image);
        sleep(10);
    }

    qInfo() << "DefaulImageThread finished";
}

}    // namespace Model
