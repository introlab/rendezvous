#ifndef VIDEO_STABILIZER_H
#define VIDEO_STABILIZER_H

#include "utils/time/Timer.h"

class VideoStabilizer
{
public:

    VideoStabilizer(int targetFps);

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

#endif //!VIDEO_STABILIZER_H