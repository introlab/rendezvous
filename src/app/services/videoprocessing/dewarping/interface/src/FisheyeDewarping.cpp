#include "FisheyeDewarping.h"

#include <display/DisplayManager.h>
#include <rendering/DewarpRenderer.h>
#include <rendering/FrameLoader.h>
#include <rendering/VertexObjectLoader.h>
#include <shaders/DewarpShader.h>
#include <shaders/RegularShader.h>
#include <models/FisheyeTexture.h>

#include <iostream>
#include <stdexcept> 

using namespace std;

FisheyeDewarping::FisheyeDewarping(int inputWidth, int inputHeight, int channels, bool isDewarping)
    : m_dewarpRenderer(nullptr),
    m_fisheyeTexture(nullptr),
    m_frameLoader(nullptr),
    m_shader(nullptr),
    m_vertexObjectLoader(nullptr)
{
    initialize(inputWidth, inputHeight, channels, isDewarping);
}

FisheyeDewarping::~FisheyeDewarping()
{
    cleanUp();
}

void FisheyeDewarping::loadFisheyeImage(unsigned char * fisheyeImage, int height, int width, int channels)
{
    m_frameLoader->load(fisheyeImage, width, height, channels);
}

int FisheyeDewarping::bindDewarpingBuffer(unsigned char * dewarpedImageBuffer, int height, int width, int channels)
{
    ImageBuffer imageBuffer(dewarpedImageBuffer, width, height, channels);
    
    int renderContextId = m_frameLoader->createRenderContext(imageBuffer.width, imageBuffer.height, imageBuffer.size);
    m_dewarpedImageBuffers.insert(pair<int, ImageBuffer>(renderContextId, imageBuffer));

    return renderContextId;
}

void FisheyeDewarping::queueDewarping(int renderContextId, DewarpingParameters& dewarpingParameters)
{
    m_dewarpingQueue.push_back(std::make_pair(renderContextId, dewarpingParameters));
}

int FisheyeDewarping::dewarpNextImage()
{
    // Calling code knows all dewarping is completed when -1 is returned
    if (m_dewarpingQueue.empty())
    {
        return -1;
    }

    // Retrieve next dewarping in queue
    auto pair = m_dewarpingQueue.front();
    int renderContextId = pair.first;
    DewarpingParameters dewarpingParameters = pair.second;
    m_dewarpingQueue.pop_front();
    
    ImageBuffer imageBuffer = m_dewarpedImageBuffers[renderContextId];
    m_frameLoader->setRenderingContext(renderContextId, imageBuffer.width, imageBuffer.height);

    // Dewarp the fisheye image with the dewarping parameters
    m_dewarpRenderer->render(*m_fisheyeTexture, dewarpingParameters);

    // Copy the dewarped image to the buffer
    m_frameLoader->unload(imageBuffer, renderContextId);

    // Calling code knows which buffer was updated based on this id
    return renderContextId;
}

void FisheyeDewarping::initialize(int inputWidth, int inputHeight, int channels, bool isDewarping)
{
    if (DisplayManager::createDisplay(1, 1) != 0)
    {
        throw runtime_error("FisheyeDewarping object initialization failed");
    }

    if (isDewarping)
        m_shader = make_shared<DewarpShader>();
    else
        m_shader = make_shared<RegularShader>();

    m_vertexObjectLoader = make_unique<VertexObjectLoader>();
    m_frameLoader = make_unique<FrameLoader>(inputWidth, inputHeight, channels, GL_BGR, SinglePBO);
    m_dewarpRenderer = make_unique<DewarpRenderer>(m_shader, *m_vertexObjectLoader);
    m_fisheyeTexture = make_unique<FisheyeTexture>(m_frameLoader->getTextureId(), glm::vec2(0, 0), glm::vec2(1, 1));    
}

void FisheyeDewarping::cleanUp()
{
    m_shader->cleanUp();
    m_frameLoader->cleanUp();
    m_vertexObjectLoader->cleanUp();
    DisplayManager::closeDisplay();
}