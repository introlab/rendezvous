#include "FisheyeDewarping.h"

#include <display/DisplayManager.h>
#include <rendering/DewarpRenderer.h>
#include <rendering/FrameLoader.h>
#include <rendering/VertexObjectLoader.h>
#include <shaders/DewarpShader.h>
#include <shaders/RegularShader.h>
#include <models/FisheyeTexture.h>

#include <iostream>

using namespace std;

FisheyeDewarping::FisheyeDewarping()
    : m_dewarpRenderer(nullptr),
    m_fisheyeTexture(nullptr),
    m_frameLoader(nullptr),
    m_shader(nullptr),
    m_vertexObjectLoader(nullptr),
    m_isInitialized(false)
{
}

FisheyeDewarping::~FisheyeDewarping()
{
    cleanUp();
}

int FisheyeDewarping::initialize(int inputWidth, int inputeight, int outputWidth, int outputHeight, int channels, bool isDewarping)
{
    if (m_isInitialized)
    {
        cerr << "FisheyeDewarping object is already initialized" << endl;
        return -1;
    }

    if (DisplayManager::createDisplay(outputWidth, outputHeight) != 0)
    {
        cerr << "FisheyeDewarping object initialization failed" << endl;
        return -1;
    }
    
    if (isDewarping)
    {
        m_shader = make_shared<DewarpShader>();
    }
    else
    {
        m_shader = make_shared<RegularShader>();
    }

    m_vertexObjectLoader = make_unique<VertexObjectLoader>();
    m_frameLoader = make_unique<FrameLoader>(inputWidth, inputeight, outputWidth, outputHeight, channels, GL_BGR, SinglePBO);
    m_dewarpRenderer = make_unique<DewarpRenderer>(m_shader, *m_vertexObjectLoader);
    m_fisheyeTexture = make_unique<FisheyeTexture>(m_frameLoader->getTextureId(), glm::vec2(0, 0), glm::vec2(1, 1));

    m_isInitialized = true;

    return 0;
}

void FisheyeDewarping::setDewarpingParameters(DewarpingParameters& dewarpingParameters)
{
    m_dewarpingParameters = dewarpingParameters;
}

void FisheyeDewarping::loadFisheyeImage(int width, int height, int channels, unsigned char * fisheyeImage)
{
    if (!m_isInitialized)
    {
        cerr << "FisheyeDewarping object is not initialized" << endl;
        return;
    }
    
    m_frameLoader->load(fisheyeImage);
}

void FisheyeDewarping::dewarpImage(int width, int height, int channels, unsigned char * dewarpedImage)
{
    if (!m_isInitialized)
    {
        cerr << "FisheyeDewarping object is not initialized" << endl;
        return;
    }
    
    m_dewarpRenderer->render(*m_fisheyeTexture, m_dewarpingParameters);
    
    DisplayManager::updateDisplay();

    m_frameLoader->unload(dewarpedImage);
}

void FisheyeDewarping::cleanUp()
{
    if (m_isInitialized)
    {
        m_shader->cleanUp();
        m_frameLoader->cleanUp();
        m_vertexObjectLoader->cleanUp();
        DisplayManager::closeDisplay();
    }
}