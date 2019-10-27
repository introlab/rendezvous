#ifndef VIDEO_STABILIZER_H
#define VIDEO_STABILIZER_H

#include "model/stream/utils/time/timer.h"

namespace Model
{

class VideoStabilizer
{
public:

    explicit VideoStabilizer(int targetFps);

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

} // Model

#endif //!VIDEO_STABILIZER_H
