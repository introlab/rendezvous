#ifndef FISHEYE_DEWARPING_H
#define FISHEYE_DEWARPING_H

#include <memory>
#include <deque>
#include <tuple>

#include <models/DewarpingObject.h>

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

enum DewarpImageCode
{
    NoDewarpingRead = -2,
    NoQueuedDewarping = -1
};

class FisheyeDewarping
{
public:

    FisheyeDewarping();
    virtual ~FisheyeDewarping();

    void loadFisheyeImage(int fisheyeContextId, unsigned char * fisheyeImage, int height, int width, int channels);
    int createFisheyeContext(int width, int height, int channels);
    int createRenderContext(int width, int height, int channels);
    void queueDewarping(int fisheyeContextId, int renderContextId, DewarpingParameters& dewarpingParameters, 
        unsigned char * dewarpedImageBuffer, int height, int width, int channels);
    void queueRendering(int fisheyeContextId, int renderContextId, 
        unsigned char * dewarpedImageBuffer, int height, int width, int channels);
    int dewarpNextImage();
    void cleanUp();

private:

    void initialize();
    
private:

    std::deque<DewarpingObject> m_dewarpingQueue;
    
    std::unique_ptr<DewarpRenderer> m_dewarpRenderer;
    std::unique_ptr<FisheyeTexture> m_fisheyeTexture;
    std::unique_ptr<FrameLoader> m_frameLoader;
    std::shared_ptr<ShaderProgram> m_shader;
    std::shared_ptr<VertexObjectLoader> m_vertexObjectLoader;

    bool m_isFirstDewarpingOfImage;
    
};

#endif //!FISHEYE_DEWARPING_H