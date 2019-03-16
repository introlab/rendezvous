#ifndef FRAME_LOADER_H
#define FRAME_LOADER_H

#include <string>
#include <vector>
#include <glad/glad.h>

struct RawModel;

enum FrameLoaderType
{
    NoPBO,
    SinglePBO,
    DoublePBO
};

class FrameLoader
{
public:

    FrameLoader(GLsizei inputWidth, GLsizei inputHeight, GLsizei outputWidth,
        GLsizei outputHeight, GLuint channelCount, GLenum pixelFormat, FrameLoaderType frameLoaderType);
    virtual ~FrameLoader();
    GLuint getTextureId();
    void load(GLubyte* inData);
    void unload(GLubyte* outData);
    void cleanUp();

private:

    void initializeUnpackTexture();
    void initializeUnpackPBOs();
    void initializePackPBOs();

    inline void updateTexture();
    inline void loadDataToTexture(GLubyte* inData);
    inline void updateTextureWithPBO();
    inline void loadDataToTextureWithPBO(GLubyte* inData);
    inline void updateOutputPBO();
    inline void unloadOutputPBO(GLubyte* outData);

private:

    GLuint m_packPBOs[2];
    GLuint m_unpackPBOs[2];
    GLubyte* m_textureData;
    GLuint m_texture;

    GLsizei m_inputWidth;
    GLsizei m_inputHeight;
    GLsizeiptr m_inputSize;

    GLsizei m_outputWidth;
    GLsizei m_outputHeight;
    GLsizeiptr m_outputSize;

    GLuint m_channelCount;
    GLenum m_pixelFormat;
    FrameLoaderType m_frameLoaderType;

};

#endif // !FRAME_LOADER_H

