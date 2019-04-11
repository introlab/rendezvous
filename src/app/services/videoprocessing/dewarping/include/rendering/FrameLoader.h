#ifndef FRAME_LOADER_H
#define FRAME_LOADER_H

#include <string>
#include <vector>
#include <tuple>
#include <glad/glad.h>

struct RawModel;
struct ImageBuffer;
struct FisheyeContext;
struct RenderContext;

enum FrameLoaderType
{
    NoPBO,
    SinglePBO,
    DoublePBO
};

class FrameLoader
{
public:

    FrameLoader(GLenum pixelFormat);
    virtual ~FrameLoader();

    GLuint getTextureId(GLuint fisheyeContextId);
    GLuint createFisheyeContext(GLsizei width, GLsizei height, GLsizei size);
    GLuint createRenderContext(GLsizei width, GLsizei height, GLsizei size);
    void setRenderingContext(GLuint renderContextId, GLsizei width, GLsizei height);
    void load(GLuint fisheyeContextId, ImageBuffer& imageBuffer, FrameLoaderType frameLoaderType);
    void updateOutput(GLuint renderContextId, ImageBuffer& imageBuffer);
    void readOutput(GLuint renderContextId, ImageBuffer& imageBuffer);
    void cleanUp();

private:

    void generatePBOs(GLuint* pbos, GLsizei size, GLenum target, GLenum usage);
    void generateFBOs(GLuint* fbos, GLuint texture);
    void generateTexture(GLubyte*& textureData, GLuint& texture, GLsizei width, GLsizei height, GLsizei size, GLenum pixelFormat);

    inline void updateTexture(GLubyte* textureData, GLsizei width, GLsizei height);
    inline void loadDataToTexture(GLubyte* textureData, GLubyte* inData, GLsizei size);
    inline void updateTextureWithPBO(GLsizei width, GLsizei height);
    inline void loadDataToTextureWithPBO(GLubyte* inData, GLsizei size);
    inline void updateOutputPBO(GLsizei width, GLsizei height);
    inline void unloadOutputPBO(GLubyte* outData, GLsizei size);

private:

    GLenum m_pixelFormat;

    std::vector<FisheyeContext> m_fisheyeContexts;

    std::vector<RenderContext> m_renderContexts;
    GLuint m_currentRenderContextId;

};

#endif // !FRAME_LOADER_H

