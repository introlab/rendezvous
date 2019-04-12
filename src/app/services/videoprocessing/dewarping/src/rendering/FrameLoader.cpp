#include <rendering/FrameLoader.h>
#include <rendering/RenderContext.h>
#include <rendering/FisheyeContext.h>
#include <models/RawModel.h>
#include <models/ImageBuffer.h>

#include <iostream>
#include <cstring>
#include <stdexcept> 

FrameLoader::FrameLoader(GLenum pixelFormat)
    : m_pixelFormat(pixelFormat),
    m_currentRenderContextId(-1)
{
}

FrameLoader::~FrameLoader()
{
    for (FisheyeContext& fisheyeContext : m_fisheyeContexts)
    {
        delete[] fisheyeContext.textureData;
        fisheyeContext.textureData = nullptr;
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

void FrameLoader::generateFBOs(GLuint* fbos, GLuint texture)
{
    glGenFramebuffers(2, fbos);

    for (int i = 0; i < 2; ++i)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, fbos[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("FrameLoader - Error when creating frame buffer object!");
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);  
}

GLuint FrameLoader::getTextureId(GLuint fisheyeContextId)
{
    FisheyeContext& fisheyeContext = m_fisheyeContexts[fisheyeContextId];
    return fisheyeContext.texture;
}

GLuint FrameLoader::createFisheyeContext(GLsizei width, GLsizei height, GLsizei size)
{
    FisheyeContext fisheyeContext;
    generateTexture(fisheyeContext.textureData, fisheyeContext.texture, width, height, size, m_pixelFormat);
    generatePBOs(fisheyeContext.pbos, size, GL_PIXEL_UNPACK_BUFFER, GL_STREAM_DRAW);

    m_fisheyeContexts.push_back(fisheyeContext);
    GLuint fisheyeContextId = m_fisheyeContexts.size() - 1;

    return fisheyeContextId;
}

GLuint FrameLoader::createRenderContext(GLsizei width, GLsizei height, GLsizei size)
{
    RenderContext renderContext;
    generateTexture(renderContext.textureData, renderContext.texture, width, height, size, m_pixelFormat);
    generateFBOs(renderContext.fbos, renderContext.texture);
    generatePBOs(renderContext.pbos, size, GL_PIXEL_PACK_BUFFER, GL_STREAM_READ);

    m_renderContexts.push_back(renderContext);
    GLuint renderContextId = m_renderContexts.size() - 1;

    return renderContextId;
}

void FrameLoader::setRenderingContext(GLuint renderContextId, GLsizei width, GLsizei height)
{
    if (m_currentRenderContextId != renderContextId)
    {
        m_currentRenderContextId = renderContextId;
        glViewport(0, 0, width, height);
    }

    RenderContext& renderContext = m_renderContexts[renderContextId];
    glBindFramebuffer(GL_FRAMEBUFFER, renderContext.fbos[renderContext.fboIndex]);
    glPixelStorei(GL_PACK_ROW_LENGTH, width);

    renderContext.fboIndex = (renderContext.fboIndex + 1) % 2;
    renderContext.pboIndex = (renderContext.pboIndex + 1) % 2;
}

void FrameLoader::load(GLuint fisheyeContextId, ImageBuffer& imageBuffer, FrameLoaderType frameLoaderType)
{
    FisheyeContext& fisheyeContext = m_fisheyeContexts[fisheyeContextId];

    glBindTexture(GL_TEXTURE_2D, fisheyeContext.texture);
    glPixelStorei(GL_PACK_ALIGNMENT, (imageBuffer.channels & 3) ? 1 : 4);

    if (frameLoaderType != NoPBO)
    {
        // Alternate between buffers
        GLuint bufferIndex = ++fisheyeContext.pboIndex;
        int nextBufferIndex = frameLoaderType == DoublePBO && bufferIndex ? (bufferIndex + 1) : bufferIndex;

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, fisheyeContext.pbos[bufferIndex % 2]);
        loadDataToTextureWithPBO(imageBuffer.image, imageBuffer.size);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, fisheyeContext.pbos[nextBufferIndex % 2]);
        updateTextureWithPBO(imageBuffer.width, imageBuffer.height);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    }
    else
    {
        loadDataToTexture(fisheyeContext.textureData, imageBuffer.image, imageBuffer.size);
        updateTexture(fisheyeContext.textureData, imageBuffer.width, imageBuffer.height);
    }

    // Required if texture GL_TEXTURE_MIN_FILTER was set to GL_LINEAR_MIPMAP_LINEAR
    glGenerateMipmap(GL_TEXTURE_2D);
}

void FrameLoader::updateOutput(GLuint renderContextId, ImageBuffer& imageBuffer)
{
    RenderContext& renderContext = m_renderContexts[renderContextId];

    glBindBuffer(GL_PIXEL_PACK_BUFFER, renderContext.pbos[renderContext.pboIndex]);
    updateOutputPBO(imageBuffer.width, imageBuffer.height);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void FrameLoader::readOutput(GLuint renderContextId, ImageBuffer& imageBuffer)
{
    RenderContext& renderContext = m_renderContexts[renderContextId];

    glBindBuffer(GL_PIXEL_PACK_BUFFER, renderContext.pbos[renderContext.pboIndex]);
    unloadOutputPBO(imageBuffer.image, imageBuffer.size);

    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void FrameLoader::cleanUp()
{
    for (FisheyeContext& fisheyeContext : m_fisheyeContexts)
    {
        glDeleteBuffers(2, fisheyeContext.pbos);
        glDeleteTextures(1, &fisheyeContext.texture);
    }

    for (RenderContext& renderContext : m_renderContexts)
    {
        glDeleteBuffers(2, renderContext.pbos);
        glDeleteFramebuffers(2, renderContext.fbos);
        glDeleteTextures(1, &renderContext.texture);
    }
}

void FrameLoader::updateTexture(GLubyte* textureData, GLsizei width, GLsizei height)
{
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, m_pixelFormat, GL_UNSIGNED_BYTE, (GLvoid*) textureData);
}

void FrameLoader::loadDataToTexture(GLubyte* textureData, GLubyte* inData, GLsizei size)
{
    memcpy(textureData, inData, size);
}

void FrameLoader::updateTextureWithPBO(GLsizei width, GLsizei height)
{
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, m_pixelFormat, GL_UNSIGNED_BYTE, 0);
}

void FrameLoader::loadDataToTextureWithPBO(GLubyte* inData, GLsizei size)
{
    // Clear previous buffer is still in use
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size, 0, GL_STREAM_DRAW);

    // Map the buffer object into client's memory
    GLubyte* dst = (GLubyte*) glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);

    if (dst)
    {
        memcpy(dst, inData, size);
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
