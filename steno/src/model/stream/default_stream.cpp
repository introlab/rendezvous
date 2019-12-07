#include "default_stream.h"

#include "model/config/config.h"
#include "model/stream/default_image_thread.h"
#include "model/stream/video/output/virtual_camera_output.h"

#include "memory"

#include "QCoreApplication"

namespace Model
{
DefaultStream::DefaultStream(std::shared_ptr<Config> config)
{
    const auto videoInputConf = config->videoOutputConfig();
    std::shared_ptr<IVideoOutput> vcOutput = std::make_shared<VirtualCameraOutput>(videoInputConf);

    const QString defaultImagePath = QCoreApplication::applicationDirPath() + "/../resources/defaultImage.jpg";
    m_defaultImageThread = std::make_unique<DefaultImageThread>(vcOutput, videoInputConf, defaultImagePath);
}

/**
 * @brief Wait that all threads are stopped. Do not call this method if you called the function DefaultStream::stop.
 */
void DefaultStream::join()
{
    m_defaultImageThread->join();
}

/**
 * @brief Start the thread that push the default image in our virtual camera.
 */
void DefaultStream::start()
{
    m_defaultImageThread->start();
}

/**
 * @brief Stop all threads.
 */
void DefaultStream::stop()
{
    m_defaultImageThread->stop();
    m_defaultImageThread->join();
}

/**
 * @brief Emit a signal that tells the default stream changed of state.
 * @param state
 */
void DefaultStream::updateState(const IStream::State &state)
{
    m_state = state;
    emit stateChanged(m_state);
}
}    // namespace Model
