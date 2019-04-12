#include "FisheyeDewarping.h"

#include <display/DisplayManager.h>
#include <rendering/DewarpRenderer.h>
#include <rendering/FrameLoader.h>
#include <rendering/VertexObjectLoader.h>
#include <models/FisheyeTexture.h>

#include <iostream>
#include <stdexcept> 

using namespace std;

FisheyeDewarping::FisheyeDewarping()
    : m_dewarpRenderer(nullptr),
    m_fisheyeTexture(nullptr),
    m_frameLoader(nullptr),
    m_shader(nullptr),
    m_vertexObjectLoader(nullptr),
    m_isFirstDewarpingOfImage(true)
{
    initialize();
}

FisheyeDewarping::~FisheyeDewarping()
{
}

void FisheyeDewarping::loadFisheyeImage(int fisheyeContextId, unsigned char * fisheyeImage, int height, int width, int channels)
{
    ImageBuffer imageBuffer(fisheyeImage, width, height, channels);
    m_frameLoader->load(fisheyeContextId, imageBuffer, SinglePBO);
}

int FisheyeDewarping::createFisheyeContext(int width, int height, int channels)
{
    return m_frameLoader->createFisheyeContext(width, height, width * height * channels);
}

int FisheyeDewarping::createRenderContext(int width, int height, int channels)
{
    return m_frameLoader->createRenderContext(width, height, width * height * channels);
}

void FisheyeDewarping::queueDewarping(int fisheyeContextId, int renderContextId, DewarpingParameters& dewarpingParameters, 
    unsigned char * dewarpedImageBuffer, int height, int width, int channels)
{
    ImageBuffer imageBuffer(dewarpedImageBuffer, width, height, channels);
    m_dewarpingQueue.push_back(DewarpingObject{fisheyeContextId, renderContextId, imageBuffer, dewarpingParameters});
}

void FisheyeDewarping::queueRendering(int fisheyeContextId, int renderContextId, 
    unsigned char * dewarpedImageBuffer, int height, int width, int channels)
{
    ImageBuffer imageBuffer(dewarpedImageBuffer, width, height, channels);
    m_dewarpingQueue.push_back(DewarpingObject{fisheyeContextId, renderContextId, imageBuffer});
}

int FisheyeDewarping::dewarpNextImage()
{
    int unloadRenderContextId;

    // Calling code knows all dewarping is completed when NoQueuedDewarping is returned
    if (m_dewarpingQueue.empty())
    {
        return NoQueuedDewarping;
    }

    // In order to take advantage of asynchronous gpu reading, 
    // we don't render and unload on the same dewarpNextImage call a queued dewarping
    if (m_isFirstDewarpingOfImage)
    {
        m_isFirstDewarpingOfImage = false;
        unloadRenderContextId = NoDewarpingRead;
    }
    else
    {
        DewarpingObject& unloadDewarpingObject = m_dewarpingQueue.front();   

        // Copy the dewarped image to the buffer
        m_frameLoader->readOutput(unloadDewarpingObject.renderContextId, unloadDewarpingObject.imageBuffer);
        unloadRenderContextId = unloadDewarpingObject.renderContextId;
        
        m_dewarpingQueue.pop_front(); 
    }

    if (m_dewarpingQueue.empty())
    {
        // All dewarping is completed, reset for next items in queue
        m_isFirstDewarpingOfImage = true;
    }
    else
    {
        // Retrieve next dewarping in queue
        DewarpingObject& renderDewarpingObject = m_dewarpingQueue.front();
        ImageBuffer& imageBuffer = renderDewarpingObject.imageBuffer;
        
        m_frameLoader->setRenderingContext(renderDewarpingObject.renderContextId, imageBuffer.width, imageBuffer.height);

        // Select the fisheye image used for dewarping or rendering
        GLuint textureId = m_frameLoader->getTextureId(renderDewarpingObject.fisheyeContextId);
        m_fisheyeTexture->texture = textureId;

        if (renderDewarpingObject.isDewarping)
        {
            // Dewarp the fisheye image with the dewarping parameters
            m_dewarpRenderer->renderDewarping(*m_fisheyeTexture, renderDewarpingObject.dewarpingParameters);
        }
        else
        {
            // Render the fisheye image without dewarping it
            m_dewarpRenderer->render(*m_fisheyeTexture);
        }
        m_frameLoader->updateOutput(renderDewarpingObject.renderContextId, imageBuffer);
    }

    return unloadRenderContextId;
}

void FisheyeDewarping::initialize()
{
    if (DisplayManager::createDisplay(1, 1) != 0)
    {
        throw runtime_error("FisheyeDewarping object initialization failed");
    }

    m_vertexObjectLoader = make_shared<VertexObjectLoader>();
    m_frameLoader = make_unique<FrameLoader>(GL_BGR);
    m_dewarpRenderer = make_unique<DewarpRenderer>(m_vertexObjectLoader);
    m_fisheyeTexture = make_unique<FisheyeTexture>(0, glm::vec2(0, 0), glm::vec2(1, 1));  
}

void FisheyeDewarping::cleanUp()
{
    m_dewarpRenderer->cleanUp();
    m_frameLoader->cleanUp();
    m_vertexObjectLoader->cleanUp();
    DisplayManager::closeDisplay();
}