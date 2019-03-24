#include <rendering/DewarpRenderer.h>
#include <rendering/VertexObjectLoader.h>
#include <models/RawModel.h>
#include <models/FisheyeTexture.h>
#include <models/DewarpingParameters.h>
#include <shaders/ShaderProgram.h>
#include <utils/Maths.h>

#include <glad/glad.h>
#include <math.h>
#include <iostream>

DewarpRenderer::DewarpRenderer(std::shared_ptr<ShaderProgram>& shader, std::shared_ptr<VertexObjectLoader>& loader)
    : m_shader(shader)
{
    float positions[] = { -1, 1, -1, -1, 1, 1, 1, -1 };
    m_quad = loader->loadToVAO(positions, 8);
    initializeGlContext();
}

DewarpRenderer::~DewarpRenderer()
{
}

void DewarpRenderer::initializeGlContext()
{
    glDisable(GL_DEPTH_TEST);
    glActiveTexture(GL_TEXTURE0);
    glClearColor(0, 0, 0, 0);
}

void DewarpRenderer::render(FisheyeTexture& fisheyeTexture, DewarpingParameters& dewarpingParameters)
{
    m_shader->start();
    glBindVertexArray(m_quad->vaoId);
    glEnableVertexAttribArray(0);

    glBindTexture(GL_TEXTURE_2D, fisheyeTexture.texture);
    glm::mat4 matrix = Maths::createTransformationMatrix(fisheyeTexture.position, fisheyeTexture.scale);
    m_shader->loadTransformation(matrix);
    m_shader->loadDewarpingParameters(dewarpingParameters);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, m_quad->vertexCount);

    glBindVertexArray(0);
    m_shader->stop();
}
