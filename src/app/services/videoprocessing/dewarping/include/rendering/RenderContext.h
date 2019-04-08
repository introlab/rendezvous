#ifndef RENDER_CONTEXT_H
#define RENDER_CONTEXT_H

#include <glad/glad.h>

struct RenderContext
{
    GLuint pbos[2];
    GLuint fbo;
    GLuint texture;
    GLubyte* textureData;
};

#endif //!RENDER_CONTEXT_H