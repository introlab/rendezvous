#include <utils/Maths.h>

#include <glm/gtc/matrix_transform.hpp>

glm::mat4 Maths::createTransformationMatrix(glm::vec2& translation, glm::vec2& scale)
{
    glm::mat4 matrix(1.0);
    glm::vec3 translation3D(translation.x, translation.y, 0);
    matrix = glm::translate(matrix, translation3D);
    glm::vec3 scale3D(scale.x, scale.y, 1);
    matrix = glm::scale(matrix, scale3D);
    return matrix;
}
