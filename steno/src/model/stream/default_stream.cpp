#include "default_stream.h"

#include "model/config/config.h"
#include "model/stream/default_image_thread.h"
#include "model/stream/video/output/virtual_camera_output.h"

#include "memory"

namespace Model
{
DefaultStream::DefaultStream(std::shared_ptr<Config> config)
{
    const auto videoInputConf = config->videoInputConfig();
    std::shared_ptr<IVideoOutput> vcOutput = std::make_shared<VirtualCameraOutput>(videoInputConf);
    m_defaultImageThread = std::make_unique<DefaultImageThread>(vcOutput, videoInputConf);
    m_defaultImageThread->attach(this);
}

DefaultStream::~DefaultStream()
{
    m_defaultImageThread->detach(this);
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
    updateState(State::Stopping);
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

void DefaultStream::updateObserver()
{
    DefaultImageThread::ThreadStatus state = m_defaultImageThread->getState();
    if (state == DefaultImageThread::ThreadStatus::CRASHED || state == DefaultImageThread::ThreadStatus::STOPPED)
    {
        updateState(State::Stopped);
    }
    else if (state == DefaultImageThread::ThreadStatus::RUNNING)
    {
        updateState(State::Started);
    }
}
}    // namespace Model
