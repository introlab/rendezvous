#include "default_image_thread.h"

#include "model/stream/video/input/image_file_reader.h"
#include "model/stream/video/output/virtual_camera_output.h"

#include <QDir>
#include <QFile>

namespace Model
{
DefaultImageThread::DefaultImageThread(std::shared_ptr<IVideoOutput> videoOutput,
                                       std::shared_ptr<VideoConfig> videoConfig, const QString defaultImagePath)
    : m_videoOutput(videoOutput)
    , m_videoConfig(videoConfig)
    , m_imageFilePath(defaultImagePath)
{
}

void DefaultImageThread::run()
{
    qInfo() << "DefaulImageThread started";
    m_state = ThreadStatus::RUNNING;
    notify();

    QDir dir(m_imageFilePath);
    dir.makeAbsolute();

    ImageFormat imgFormat = static_cast<ImageFormat>(m_videoConfig->value(VideoConfig::IMAGE_FORMAT).toInt());
    ImageFileReader imageFileReader(dir.absolutePath().toStdString(), imgFormat);
    imageFileReader.open();
    const Image& image = imageFileReader.readImage();
    imageFileReader.close();

    m_videoOutput->open();
    while (!isAbortRequested())
    {
        try
        {
            m_videoOutput->writeImage(image);
            sleep(100);
        }
        catch (const std::exception& e)
        {
            std::cout << "Error in default image thread : " << e.what() << std::endl;
            m_state = ThreadStatus::CRASHED;
            break;
        }
    }

    m_videoOutput->close();
    qInfo() << "DefaulImageThread finished";
    if (m_state != ThreadStatus::CRASHED)
    {
        m_state = ThreadStatus::STOPPED;
    }
    notify();
}
}    // namespace Model
