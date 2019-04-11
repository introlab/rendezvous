#ifndef FISHEYE_CONTEXT_H
#define FISHEYE_CONTEXT_H

#include <glad/glad.h>

struct FisheyeContext
{
    FisheyeContext();

    GLuint pbos[2];
    GLuint pboIndex;
    GLuint texture;
    GLubyte* textureData;
};

#endif //!FISHEYE_CONTEXT_H