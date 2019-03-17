#ifndef VERTEX_OBJECT_LOADER_H
#define VERTEX_OBJECT_LOADER_H

#include <memory>
#include <string>
#include <vector>
#include <glad/glad.h>

struct RawModel;

class VertexObjectLoader
{
public:

    virtual ~VertexObjectLoader();
    std::unique_ptr<RawModel> loadToVAO(float* positions, GLint size);
    void cleanUp();

protected:

    GLuint createVAO();
    void storeDataInAttributeList(GLuint attributeNumber, GLint attributeSize, float* data, GLint size);
    void unbindVAO();

protected:

    std::vector<GLuint> m_vaos;
    std::vector<GLuint> m_vbos;

};

#endif // !VERTEX_OBJECT_LOADER_H

