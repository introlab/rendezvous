#include <models/FisheyeTexture.h>

FisheyeTexture::FisheyeTexture(GLuint texture, glm::vec2& position, glm::vec2& scale)
    : texture(texture),
    position(position),
    scale(scale)
{
}

FisheyeTexture::FisheyeTexture(GLuint texture, glm::vec2&& position, glm::vec2&& scale)
    : texture(texture),
    position(position),
    scale(scale)
{
}
