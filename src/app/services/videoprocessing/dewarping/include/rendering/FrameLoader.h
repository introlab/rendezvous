#ifndef FRAME_LOADER_H
#define FRAME_LOADER_H

#include <string>
#include <vector>
#include <tuple>
#include <glad/glad.h>

struct RawModel;
struct ImageBuffer;

enum FrameLoaderType
{
    NoPBO,
    SinglePBO,
    DoublePBO
};

class FrameLoader
{
public:

    FrameLoader(GLsizei inputWidth, GLsizei inputHeight, GLuint channels, 
        GLenum pixelFormat, FrameLoaderType frameLoaderType);
    virtual ~FrameLoader();

    GLuint getTextureId();
    GLuint createPackPBOs(GLsizei size);
    void load(GLubyte* inData, GLsizei width, GLsizei height, GLuint channels);
    void unload(ImageBuffer& imageBuffer, GLuint bufferId);
    void cleanUp();

private:

    void initializeUnpackTexture();
    void initializeUnpackPBOs();
    void initializePackPBOs(GLuint* packPBOs, GLsizei size);

    inline void updateTexture();
    inline void loadDataToTexture(GLubyte* inData);
    inline void updateTextureWithPBO();
    inline void loadDataToTextureWithPBO(GLubyte* inData);
    inline void updateOutputPBO(GLsizei width, GLsizei height);
    inline void unloadOutputPBO(GLubyte* outData, GLsizei size);

private:

    std::vector<GLuint*> m_packPBOsVector;

    GLuint m_unpackPBOs[2];
    GLubyte* m_textureData;
    GLuint m_texture;

    GLsizei m_inputWidth;
    GLsizei m_inputHeight;
    GLsizeiptr m_inputSize;

    GLuint m_channels;
    GLenum m_pixelFormat;
    FrameLoaderType m_frameLoaderType;

};

#endif // !FRAME_LOADER_H

