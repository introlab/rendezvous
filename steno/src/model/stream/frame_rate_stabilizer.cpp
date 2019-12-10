#include "frame_rate_stabilizer.h"

#include <iostream>
#include <thread>

#include "model/stream/utils/math/helpers.h"

namespace Model
{
FrameRateStabilizer::FrameRateStabilizer(int targetFps)
    : frameTimeTargetMs_(1000 / targetFps)
    , frameTimeDeltaMs_(0)
    , frameTimeMs_(0)
    , frameTimeModifiedTargetMs_(frameTimeTargetMs_)
{
}

void FrameRateStabilizer::startFrame()
{
    timer_.reset();
}

void FrameRateStabilizer::endFrame()
{
    frameTimeMs_ = static_cast<int>(timer_.getElapsedTime<std::chrono::milliseconds>());

    // If the frame took less time than required, we need to sleep to have the right FPS
    if (frameTimeMs_ < frameTimeTargetMs_)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(frameTimeTargetMs_ - frameTimeMs_));
        frameTimeMs_ = static_cast<int>(
            timer_.getElapsedTime<std::chrono::milliseconds>());    // Recalculate as sleep is not exact
    }

    // TODO: Will be removed or fixed soon

    // // Acumulate the error on the frames
    // frameTimeDeltaMs_ += (frameTimeTargetMs_ - frameTimeMs_);

    // // If the frame time delta is too big, we want to gradually come back to our target
    // if (std::abs(frameTimeDeltaMs_) < frameTimeTargetMs_ / 6)
    // {
    //     frameTimeModifiedTargetMs_ = frameTimeTargetMs_ + frameTimeDeltaMs_;
    // }
    // else
    // {
    //     frameTimeModifiedTargetMs_ = frameTimeTargetMs_ + (frameTimeTargetMs_ / 6) * math::sign(frameTimeDeltaMs_);
    // }
}

int FrameRateStabilizer::getLastFrameTimeMs()
{
    return frameTimeMs_;
}
}    // namespace Model
