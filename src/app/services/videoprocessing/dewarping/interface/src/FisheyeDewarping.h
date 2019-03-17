#ifndef FISHEYE_DEWARPING_H
#define FISHEYE_DEWARPING_H

#include <memory>
#include <models/DewarpingParameters.h>

class DewarpRenderer;
class FisheyeTexture;
class FrameLoader;
class ShaderProgram;
class VertexObjectLoader;

class FisheyeDewarping
{
public:

    FisheyeDewarping();
    virtual ~FisheyeDewarping();

    int initialize(int inputWidth, int inputHeight, int outputWidth, int outputHeight, int channels, bool isDewarping = true);
    void setDewarpingParameters(DewarpingParameters& dewarpingParameters);
    void loadFisheyeImage(int width, int height, int channels, unsigned char * fisheyeImage);
    void dewarpImage(int width, int height, int channels, unsigned char * dewarpedImage);

private:

    void cleanUp();

private:

    DewarpingParameters m_dewarpingParameters;

    std::unique_ptr<DewarpRenderer> m_dewarpRenderer;
    std::unique_ptr<FisheyeTexture> m_fisheyeTexture;
    std::unique_ptr<FrameLoader> m_frameLoader;
    std::shared_ptr<ShaderProgram> m_shader;
    std::unique_ptr<VertexObjectLoader> m_vertexObjectLoader;
    
    bool m_isInitialized;
    
};

#endif //!FISHEYE_DEWARPING_H