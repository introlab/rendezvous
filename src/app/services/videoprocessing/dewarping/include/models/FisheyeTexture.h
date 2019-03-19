#ifndef FISHEYE_TEXTURE_H
#define FISHEYE_TEXTURE_H

#include <glm/glm.hpp>
#include <glad/glad.h>

struct FisheyeTexture
{
public:

    FisheyeTexture(GLuint texture, glm::vec2 position, glm::vec2 scale);

    GLuint texture;
    glm::vec2 position;
    glm::vec2 scale;

};

#endif // !FISHEYE_TEXTURE_H

