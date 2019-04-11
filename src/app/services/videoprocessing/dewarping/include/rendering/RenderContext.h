#ifndef RENDER_CONTEXT_H
#define RENDER_CONTEXT_H

#include <glad/glad.h>

struct RenderContext
{
    RenderContext();

    GLuint pbos[2];
    GLuint fbos[2];
    GLuint pboIndex;
    GLuint fboIndex;
    GLuint texture;
    GLubyte* textureData;
};

#endif //!RENDER_CONTEXT_H