#include <rendering/DewarpRenderer.h>
#include <rendering/VertexObjectLoader.h>
#include <models/RawModel.h>
#include <models/FisheyeTexture.h>
#include <models/DewarpingParameters.h>
#include <shaders/DewarpShader.h>
#include <shaders/RegularShader.h>
#include <shaders/ShaderProgram.h>
#include <utils/Maths.h>

#include <glad/glad.h>
#include <math.h>
#include <iostream>

DewarpRenderer::DewarpRenderer(std::shared_ptr<VertexObjectLoader>& loader)
    : m_dewarpShader(std::make_shared<DewarpShader>()),
    m_regularShader(std::make_shared<RegularShader>())
{
    float positions[] = { -1, 1, -1, -1, 1, 1, 1, -1 };
    m_quad = loader->loadToVAO(positions, 8);
    initializeGlContext();
}

DewarpRenderer::~DewarpRenderer()
{
}

void DewarpRenderer::renderDewarping(FisheyeTexture& fisheyeTexture, DewarpingParameters& dewarpingParameters)
{
    auto shaderProgram = std::dynamic_pointer_cast<ShaderProgram>(m_dewarpShader);
    
    activateShader(shaderProgram);

    glBindTexture(GL_TEXTURE_2D, fisheyeTexture.texture);
    glm::mat4 matrix = Maths::createTransformationMatrix(fisheyeTexture.position, fisheyeTexture.scale);
    m_dewarpShader->loadTransformation(matrix);
    m_dewarpShader->loadDewarpingParameters(dewarpingParameters);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, m_quad->vertexCount);

    deactivateShader(shaderProgram);
}

void DewarpRenderer::render(FisheyeTexture& fisheyeTexture)
{
    auto shaderProgram = std::dynamic_pointer_cast<ShaderProgram>(m_regularShader);
    
    activateShader(shaderProgram);

    glBindTexture(GL_TEXTURE_2D, fisheyeTexture.texture);
    glm::mat4 matrix = Maths::createTransformationMatrix(fisheyeTexture.position, fisheyeTexture.scale);
    m_regularShader->loadTransformation(matrix);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, m_quad->vertexCount);

    deactivateShader(shaderProgram);
}

void DewarpRenderer::cleanUp()
{
    m_dewarpShader->cleanUp();
    m_regularShader->cleanUp();
}

void DewarpRenderer::initializeGlContext()
{
    glDisable(GL_DEPTH_TEST);
    glActiveTexture(GL_TEXTURE0);
    glClearColor(0, 0, 0, 0);
}

void DewarpRenderer::activateShader(std::shared_ptr<ShaderProgram>& shaderProgram)
{
    shaderProgram->start();
    glBindVertexArray(m_quad->vaoId);
    glEnableVertexAttribArray(0);
}

void DewarpRenderer::deactivateShader(std::shared_ptr<ShaderProgram>& shaderProgram)
{
    glBindVertexArray(0);
    shaderProgram->stop();
}
