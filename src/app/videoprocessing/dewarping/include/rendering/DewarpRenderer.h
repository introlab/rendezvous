#ifndef DEWARP_RENDERER_H
#define DEWARP_RENDERER_H

#include <memory>

class VertexObjectLoader;
class ShaderProgram;

struct RawModel;
struct FisheyeTexture;
struct DewarpingParameters;

class DewarpRenderer
{
public:

    DewarpRenderer(std::shared_ptr<ShaderProgram> shader, VertexObjectLoader& loader);
    virtual ~DewarpRenderer();

    void render(FisheyeTexture& fisheyeTexture, DewarpingParameters& dewarpingParameters);

private:

    void initializeGlContext();

private:

    std::unique_ptr<RawModel> m_quad;
    std::shared_ptr<ShaderProgram> m_shader;

};

#endif // !DEWARP_RENDERER_H

