#include <rendering/FrameLoader.h>
#include <rendering/RenderContext.h>
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
    m_frameLoaderType(frameLoader360Type),
    m_currentRenderContextId(-1)
{
    generateTexture(m_textureData, m_texture, m_inputWidth, m_inputHeight, m_inputSize, m_pixelFormat);
    generatePBOs(m_unpackPBOs, m_inputSize, GL_PIXEL_UNPACK_BUFFER, GL_STREAM_DRAW);
    glPixelStorei(GL_PACK_ALIGNMENT, (m_channels & 3) ? 1 : 4);
}

FrameLoader::~FrameLoader()
{
    if (m_textureData)
    {
        delete[] m_textureData;
        m_textureData = nullptr;
    }

    for (RenderContext& renderContext : m_renderContexts)
    {
        delete[] renderContext.textureData;
        renderContext.textureData = nullptr;
    }
}

void FrameLoader::generateTexture(GLubyte*& textureData, GLuint& texture, 
    GLsizei width, GLsizei height, GLsizei size, GLenum pixelFormat)
{
    if (width % 4 != 0 || height % 4 != 0)
        throw std::invalid_argument("FrameLoader - texture width and height must be multiples of 4");
    
    textureData = new GLubyte[size];
    memset(textureData, 0, size);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, pixelFormat, GL_UNSIGNED_BYTE, (GLvoid*) textureData);
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

void FrameLoader::generatePBOs(GLuint* pbos, GLsizei size, GLenum target, GLenum usage)
{
    glGenBuffers(2, pbos);
    glBindBuffer(target, pbos[0]);
    glBufferData(target, size, 0, usage);
    glBindBuffer(target, pbos[1]);
    glBufferData(target, size, 0, usage);
    glBindBuffer(target, 0);
}

void FrameLoader::generateFBO(GLuint& fbo, GLuint texture)
{
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("FrameLoader - Error when creating frame buffer object!");

    glBindFramebuffer(GL_FRAMEBUFFER, 0);  
}

GLuint FrameLoader::getTextureId()
{
    return m_texture;
}

GLuint FrameLoader::createRenderContext(GLsizei width, GLsizei height, GLsizei size)
{
    RenderContext renderContext;
    generateTexture(renderContext.textureData, renderContext.texture, width, height, size, m_pixelFormat);
    generateFBO(renderContext.fbo, renderContext.texture);
    generatePBOs(renderContext.pbos, size, GL_PIXEL_PACK_BUFFER, GL_STREAM_READ);

    m_renderContexts.push_back(renderContext);
    GLuint renderContextId = m_renderContexts.size() - 1;

    return renderContextId;
}

void FrameLoader::setRenderingContext(GLuint renderContextId, GLsizei width, GLsizei height)
{
    if (m_currentRenderContextId == renderContextId)
        return;

    m_currentRenderContextId = renderContextId;
    RenderContext& renderContext = m_renderContexts[renderContextId];
    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, renderContext.fbo);
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

void FrameLoader::unload(ImageBuffer& imageBuffer, GLuint renderContextId)
{
    RenderContext& renderContext = m_renderContexts[renderContextId];

    glPixelStorei(GL_PACK_ROW_LENGTH, imageBuffer.width);

    static unsigned int bufferIndex = -1;

    // Alternate between buffers
    bufferIndex = (bufferIndex + 1);
    int nextBufferIndex = m_frameLoaderType == DoublePBO && bufferIndex ? (bufferIndex + 1) : bufferIndex;

    glBindBuffer(GL_PIXEL_PACK_BUFFER, renderContext.pbos[bufferIndex % 2]);
    updateOutputPBO(imageBuffer.width, imageBuffer.height);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, renderContext.pbos[nextBufferIndex % 2]);
    unloadOutputPBO(imageBuffer.image, imageBuffer.size);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void FrameLoader::cleanUp()
{
    glDeleteTextures(1, &m_texture);
    glDeleteBuffers(2, m_unpackPBOs);

    for (RenderContext& renderContext : m_renderContexts)
    {
        glDeleteBuffers(2, renderContext.pbos);
        glDeleteFramebuffers(1, &renderContext.fbo);
        glDeleteTextures(1, &renderContext.texture);
    }
}

void FrameLoader::updateTexture()
{
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_inputWidth, m_inputHeight, m_pixelFormat, GL_UNSIGNED_BYTE, (GLvoid*) m_textureData);
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
