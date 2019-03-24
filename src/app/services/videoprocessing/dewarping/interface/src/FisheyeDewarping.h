#ifndef FISHEYE_DEWARPING_H
#define FISHEYE_DEWARPING_H

#include <memory>
#include <queue>
#include <map>
#include <utility> 
#include <models/DewarpingParameters.h>
#include <models/ImageBuffer.h>

class DewarpRenderer;
class FisheyeTexture;
class FrameLoader;
class ShaderProgram;
class VertexObjectLoader;

/*
 * This class is the main interface between the c++ and python code.
 * Functions which takes a 3 dimentional arrays start with height instead of width,
 * because numpy arrays have row and col inverted from OpenGL arrays
 */

class FisheyeDewarping
{
public:

    FisheyeDewarping(int inputWidth, int inputHeight, int channels, bool isDewarping = true);
    virtual ~FisheyeDewarping();

    void loadFisheyeImage(unsigned char * fisheyeImage, int height, int width, int channels);
    int createRenderContext(int width, int height, int channels);
    void queueDewarping(int renderContextId, DewarpingParameters& dewarpingParameters, 
        unsigned char * dewarpedImageBuffer, int height, int width, int channels);
    int dewarpNextImage();

private:

    void initialize(int inputWidth, int inputHeight, int channels, bool isDewarping);
    void cleanUp();

private:

    std::deque<std::tuple<int, DewarpingParameters, ImageBuffer>> m_dewarpingQueue;

    std::unique_ptr<DewarpRenderer> m_dewarpRenderer;
    std::unique_ptr<FisheyeTexture> m_fisheyeTexture;
    std::unique_ptr<FrameLoader> m_frameLoader;
    std::shared_ptr<ShaderProgram> m_shader;
    std::shared_ptr<VertexObjectLoader> m_vertexObjectLoader;
    
};

#endif //!FISHEYE_DEWARPING_H