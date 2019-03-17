#include <rendering/FrameLoader.h>
#include <models/RawModel.h>
#include <models/ImageBuffer.h>

#include <iostream>
#include <cstring>
#include <stdexcept> 

FrameLoader::FrameLoader(GLsizei inputWidth, GLsizei inputHeight, 
    GLuint channels, GLenum pixelFormat, FrameLoaderType frameLoader360Type)
    : m_inputWidth(inputWidth),
    m_inputHeight(inputHeight),
    m_inputSize(inputWidth * inputHeight * channels),
    m_channels(channels),
    m_pixelFormat(pixelFormat),
    m_frameLoaderType(frameLoader360Type)
{
    initializeUnpackTexture();
    initializeUnpackPBOs();
    glPixelStorei(GL_PACK_ALIGNMENT, (m_channels & 3) ? 1 : 4);
}

FrameLoader::~FrameLoader()
{
    if (m_textureData)
    {
        delete[] m_textureData;
        m_textureData = nullptr;
    }

    for (GLuint * packPBOs : m_packPBOsVector)
    {
        delete[] packPBOs;
    }
}

void FrameLoader::initializeUnpackTexture()
{
    m_textureData = new GLubyte[m_inputSize];
    memset(m_textureData, 0, m_inputSize);

    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_inputWidth, m_inputHeight, 0, m_pixelFormat, GL_UNSIGNED_BYTE, (GLvoid*) m_textureData);
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

void FrameLoader::initializeUnpackPBOs()
{
    glGenBuffers(2, m_unpackPBOs);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_unpackPBOs[0]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_inputSize, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_unpackPBOs[1]);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_inputSize, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void FrameLoader::initializePackPBOs(GLuint* packPBOs, GLsizei size)
{
    glGenBuffers(2, packPBOs);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, packPBOs[0]);
    glBufferData(GL_PIXEL_PACK_BUFFER, size, 0, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, packPBOs[1]);
    glBufferData(GL_PIXEL_PACK_BUFFER, size, 0, GL_STREAM_READ);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

GLuint FrameLoader::getTextureId()
{
    return m_texture;
}

GLuint FrameLoader::createPackPBOs(GLsizei size)
{
    GLuint * packPBOs = new GLuint[2];
    initializePackPBOs(packPBOs, size);
    
    m_packPBOsVector.push_back(packPBOs);
    GLuint bufferId = m_packPBOsVector.size() - 1;

    return bufferId;
}

void FrameLoader::load(GLubyte* inData, GLsizei width, GLsizei height, GLuint channels)
{
    if (m_inputWidth != width || m_inputHeight != height || m_channels != channels)
        throw std::invalid_argument("FrameLoader - Image size does not match size specified on initialization");

    glBindTexture(GL_TEXTURE_2D, m_texture);

    if (m_frameLoaderType != NoPBO)
    {
        static int bufferIndex = -1;

        // Alternate between buffers
        bufferIndex = (bufferIndex + 1);
        int nextBufferIndex = m_frameLoaderType == DoublePBO && bufferIndex ? (bufferIndex + 1) : bufferIndex;

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_unpackPBOs[bufferIndex % 2]);
        loadDataToTextureWithPBO(inData);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_unpackPBOs[nextBufferIndex % 2]);
        updateTextureWithPBO();

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        loadDataToTexture(inData);
        updateTexture();
    }

    // Required if texture GL_TEXTURE_MIN_FILTER was set to GL_LINEAR_MIPMAP_LINEAR
    glGenerateMipmap(GL_TEXTURE_2D);
}

void FrameLoader::unload(ImageBuffer& imageBuffer, GLuint bufferId)
{
    // Set the framebuffer to read
    glReadBuffer(GL_FRONT);

    GLuint * packPBOs = m_packPBOsVector[bufferId];

    glPixelStorei(GL_PACK_ROW_LENGTH, imageBuffer.width);

    static unsigned int bufferIndex = -1;

    // Alternate between buffers
    bufferIndex = (bufferIndex + 1);
    int nextBufferIndex = m_frameLoaderType == DoublePBO && bufferIndex ? (bufferIndex + 1) : bufferIndex;

    glBindBuffer(GL_PIXEL_PACK_BUFFER, packPBOs[bufferIndex % 2]);
    updateOutputPBO(imageBuffer.width, imageBuffer.height);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, packPBOs[nextBufferIndex % 2]);
    unloadOutputPBO(imageBuffer.image, imageBuffer.size);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void FrameLoader::cleanUp()
{
    glDeleteTextures(1, &m_texture);
    glDeleteBuffers(2, m_unpackPBOs);

    for (GLuint * packPBOs : m_packPBOsVector)
    {
        glDeleteBuffers(2, packPBOs);
    }
}

void FrameLoader::updateTexture()
{
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_inputWidth, m_inputHeight, m_pixelFormat, GL_UNSIGNED_BYTE, (GLvoid*)m_textureData);
}

void FrameLoader::loadDataToTexture(GLubyte* inData)
{
    memcpy(m_textureData, inData, m_inputSize);
}

void FrameLoader::updateTextureWithPBO()
{
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_inputWidth, m_inputHeight, m_pixelFormat, GL_UNSIGNED_BYTE, 0);
}

void FrameLoader::loadDataToTextureWithPBO(GLubyte* inData)
{
    // Clear previous buffer is still in use
    glBufferData(GL_PIXEL_UNPACK_BUFFER, m_inputSize, 0, GL_STREAM_DRAW);

    // Map the buffer object into client's memory
    GLubyte* dst = (GLubyte*) glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);

    if (dst)
    {
        memcpy(dst, inData, m_inputSize);
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    }
}

void FrameLoader::updateOutputPBO(GLsizei width, GLsizei height)
{
    glReadPixels(0, 0, width, height, m_pixelFormat, GL_UNSIGNED_BYTE, 0);
}

void FrameLoader::unloadOutputPBO(GLubyte* outData, GLsizei size)
{
    GLubyte* src = (GLubyte*) glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

    if (src)
    {
        memcpy(outData, src, size);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    }
}
