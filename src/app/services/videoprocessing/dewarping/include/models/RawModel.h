#ifndef RAW_MODEL_H
#define RAW_MODEL_H

#include <glad/glad.h>

struct RawModel
{
public:

    RawModel(GLuint vaoId, int vertexCount);

    GLuint vaoId;
    int vertexCount;

};

#endif // !RAW_MODEL_H

