#include <rendering/VertexObjectLoader.h>
#include <models/RawModel.h>
#include <stb/stb_image.h>

#include <iostream>


VertexObjectLoader::~VertexObjectLoader()
{
}

std::unique_ptr<RawModel> VertexObjectLoader::loadToVAO(float* positions, GLint size)
{
    GLuint vaoID = createVAO();
    storeDataInAttributeList(0, 2, positions, size);
    unbindVAO();
    return std::make_unique<RawModel>(vaoID, size / 2);
}

void VertexObjectLoader::cleanUp()
{
    for (GLuint vao : m_vaos)
    {
        glDeleteVertexArrays(1, &vao);
    }
    for (GLuint vbo : m_vbos)
    {
        glDeleteBuffers(1, &vbo);
    }
}

GLuint VertexObjectLoader::createVAO()
{
    GLuint vaoID;
    glGenVertexArrays(1, &vaoID);
    m_vaos.push_back(vaoID);
    glBindVertexArray(vaoID);
    return vaoID;
}

void VertexObjectLoader::storeDataInAttributeList(GLuint attributeNumber, GLint attributeSize, float* data, GLint size)
{
    GLuint vboID;
    glGenBuffers(1, &vboID);
    m_vbos.push_back(vboID);
    glBindBuffer(GL_ARRAY_BUFFER, vboID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(data) * size, data, GL_STATIC_DRAW);
    glVertexAttribPointer(attributeNumber, attributeSize, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VertexObjectLoader::unbindVAO()
{
    glBindVertexArray(0);
}
