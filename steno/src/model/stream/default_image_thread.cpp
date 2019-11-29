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
    m_state = ThreadStatus::RUNNING;
    notify();

    QFile file(":/defaultImage.jpg");
    if (!file.exists())
    {
        qCritical() << file.symLinkTarget() << "does not exists.";
        m_state = ThreadStatus::CRASHED;
        notify();
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
    m_state = ThreadStatus::STOPPED;
    notify();
}

/**
 * @brief Attach an object to the notifications send by DefaultImageThread.
 * @param observer - object to attach to notifications.
 */
void DefaultImageThread::attach(IObserver* observer)
{
    if (observer != nullptr)
    {
        m_subscribers.push_back(observer);
    }
}

/**
 * @brief Remove an observer from the list of subscribers.
 * @param observer - object to remove from the notifications list.
 */
void DefaultImageThread::detach(IObserver* observer)
{
    for (int index = 0; static_cast<size_t>(index) < m_subscribers.size(); ++index)
    {
        if (m_subscribers.at(static_cast<size_t>(index)) == observer)
        {
            m_subscribers.erase(m_subscribers.begin() + index);
        }
    }
}

/**
 * @brief Notify all observers that the state of DefaultImageThread changed.
 */
void DefaultImageThread::notify()
{
    for (auto observer : m_subscribers)
    {
        observer->updateObserver();
    }
}

}    // namespace Model
