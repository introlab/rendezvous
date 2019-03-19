#ifndef MATHS_H
#define MATHS_H

#include <glm/glm.hpp>

class Maths
{
public:

    static glm::mat4 createTransformationMatrix(glm::vec2& translation, glm::vec2& scale);

private:
    
    Maths();

};

#endif // !MATHS_H

