#include "FisheyeDewarping.h"

#include <display/DisplayManager.h>
#include <rendering/DewarpRenderer.h>
#include <rendering/FrameLoader.h>
#include <rendering/VertexObjectLoader.h>
#include <models/FisheyeTexture.h>

#include <iostream>
#include <stdexcept> 

using namespace std;

FisheyeDewarping::FisheyeDewarping(int inputWidth, int inputHeight, int channels)
    : m_dewarpRenderer(nullptr),
    m_fisheyeTexture(nullptr),
    m_frameLoader(nullptr),
    m_shader(nullptr),
    m_vertexObjectLoader(nullptr)
{
    initialize(inputWidth, inputHeight, channels);
}

FisheyeDewarping::~FisheyeDewarping()
{
}

void FisheyeDewarping::loadFisheyeImage(unsigned char * fisheyeImage, int height, int width, int channels)
{
    m_frameLoader->load(fisheyeImage, width, height, channels);
}

int FisheyeDewarping::createRenderContext(int width, int height, int channels)
{
    return m_frameLoader->createRenderContext(width, height, width * height * channels);
}

void FisheyeDewarping::queueDewarping(int renderContextId, DewarpingParameters& dewarpingParameters, 
    unsigned char * dewarpedImageBuffer, int height, int width, int channels)
{
    ImageBuffer imageBuffer(dewarpedImageBuffer, width, height, channels);
    m_dewarpingQueue.push_back(DewarpingObject{renderContextId, imageBuffer, dewarpingParameters});
}

void FisheyeDewarping::queueRendering(int renderContextId, unsigned char * dewarpedImageBuffer, int height, int width, int channels)
{
    ImageBuffer imageBuffer(dewarpedImageBuffer, width, height, channels);
    m_dewarpingQueue.push_back(DewarpingObject{renderContextId, imageBuffer});
}

int FisheyeDewarping::dewarpNextImage()
{
    // Calling code knows all dewarping is completed when -1 is returned
    if (m_dewarpingQueue.empty())
    {
        return NoQueuedDewarping;
    }

    // Retrieve next dewarping in queue
    DewarpingObject dewarpingObject = m_dewarpingQueue.front();
    m_dewarpingQueue.pop_front();

    int renderContextId = dewarpingObject.renderContextId;
    ImageBuffer& imageBuffer = dewarpingObject.imageBuffer;
    
    m_frameLoader->setRenderingContext(renderContextId, imageBuffer.width, imageBuffer.height);

    if (dewarpingObject.isDewarping)
    {
        // Dewarp the fisheye image with the dewarping parameters
        m_dewarpRenderer->renderDewarping(*m_fisheyeTexture, dewarpingObject.dewarpingParameters);
    }
    else
    {
        m_dewarpRenderer->render(*m_fisheyeTexture);
    }

    // Copy the dewarped image to the buffer
    m_frameLoader->unload(imageBuffer, renderContextId);

    // Calling code knows which buffer was updated based on this id
    return renderContextId;
}

void FisheyeDewarping::initialize(int inputWidth, int inputHeight, int channels)
{
    if (DisplayManager::createDisplay(1, 1) != 0)
    {
        throw runtime_error("FisheyeDewarping object initialization failed");
    }

    m_vertexObjectLoader = make_shared<VertexObjectLoader>();
    m_frameLoader = make_unique<FrameLoader>(inputWidth, inputHeight, channels, GL_BGR, SinglePBO);
    m_dewarpRenderer = make_unique<DewarpRenderer>(m_vertexObjectLoader);
    m_fisheyeTexture = make_unique<FisheyeTexture>(m_frameLoader->getTextureId(), glm::vec2(0, 0), glm::vec2(1, 1));    
}

void FisheyeDewarping::cleanUp()
{
    m_dewarpRenderer->cleanUp();
    m_frameLoader->cleanUp();
    m_vertexObjectLoader->cleanUp();
    DisplayManager::closeDisplay();
}