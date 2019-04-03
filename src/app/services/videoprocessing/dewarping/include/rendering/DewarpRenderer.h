#ifndef DEWARP_RENDERER_H
#define DEWARP_RENDERER_H

#include <memory>

class VertexObjectLoader;
class DewarpShader;
class RegularShader;
class ShaderProgram;

struct RawModel;
struct FisheyeTexture;
struct DewarpingParameters;

class DewarpRenderer
{
public:

    DewarpRenderer(std::shared_ptr<VertexObjectLoader>& loader);
    virtual ~DewarpRenderer();

    void renderDewarping(FisheyeTexture& fisheyeTexture, DewarpingParameters& dewarpingParameters);
    void render(FisheyeTexture& fisheyeTexture);
    void cleanUp();

private:

    void initializeGlContext();
    inline void activateShader(std::shared_ptr<ShaderProgram>& shaderProgram);
    inline void deactivateShader(std::shared_ptr<ShaderProgram>& shaderProgram);

private:

    std::unique_ptr<RawModel> m_quad;
    std::shared_ptr<DewarpShader> m_dewarpShader;
    std::shared_ptr<RegularShader> m_regularShader;

};

#endif // !DEWARP_RENDERER_H

