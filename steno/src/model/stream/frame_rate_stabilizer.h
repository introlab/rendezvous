#ifndef FRAME_RATE__STABILIZER_H
#define FRAME_RATE__STABILIZER_H

#include "model/stream/utils/time/timer.h"

namespace Model
{
class FrameRateStabilizer
{
   public:
    explicit FrameRateStabilizer(int targetFps);

    void startFrame();
    void endFrame();
    int getLastFrameTimeMs();

   private:
    int frameTimeTargetMs_;
    int frameTimeDeltaMs_;
    int frameTimeMs_;
    int frameTimeModifiedTargetMs_;

    Timer timer_;
};

}    // namespace Model

#endif    //! FRAME_RATE__STABILIZER_H
