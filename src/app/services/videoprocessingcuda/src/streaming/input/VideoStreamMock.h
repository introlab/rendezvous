#ifndef VIDEO_STREAM_MOCK_H
#define VIDEO_STREAM_MOCK_H

#include <string>

#include "streaming/input/IVideoStream.h"
#include "utils/images/stb/stb_image.h"

class VideoStreamMock : public IVideoStream
{
public:

    VideoStreamMock(const std::string& imageFilePath);

    bool copyFrameData(const Image& image) override;
    bool getResolution(Dim3<int>& dim) override;

private:

    bool loadImage(const char* fileName, stbi_uc*& data, int& width, int& height, int& channels, int desiredChannels = 0) const;

    Image image_;

};

#endif // !VIDEO_STREAM_MOCK_H
