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
    m_dewarpingQueue.push_back(std::make_tuple(renderContextId, dewarpingParameters, imageBuffer, true));
}

void FisheyeDewarping::queueRendering(int renderContextId, unsigned char * dewarpedImageBuffer, int height, int width, int channels)
{
    ImageBuffer imageBuffer(dewarpedImageBuffer, width, height, channels);
    m_dewarpingQueue.push_back(std::make_tuple(renderContextId, DewarpingParameters{}, imageBuffer, false));
}

int FisheyeDewarping::dewarpNextImage()
{
    // Calling code knows all dewarping is completed when -1 is returned
    if (m_dewarpingQueue.empty())
    {
        return -1;
    }

    // Retrieve next dewarping in queue
    auto tuple = m_dewarpingQueue.front();
    int renderContextId = std::get<0>(tuple);
    DewarpingParameters& dewarpingParameters = std::get<1>(tuple);
    ImageBuffer& imageBuffer = std::get<2>(tuple);
    bool isDewarping = std::get<3>(tuple);
    m_dewarpingQueue.pop_front();
    
    m_frameLoader->setRenderingContext(renderContextId, imageBuffer.width, imageBuffer.height);

    if (isDewarping)
    {
        // Dewarp the fisheye image with the dewarping parameters
        m_dewarpRenderer->renderDewarping(*m_fisheyeTexture, dewarpingParameters);
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