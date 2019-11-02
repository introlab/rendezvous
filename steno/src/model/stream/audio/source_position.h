#ifndef SOURCE_POSITION_H
#define SOURCE_POSITION_H

#include <QJsonValue>

namespace Model
{
struct SourcePosition
{
    SourcePosition(float azimuth, float elevation);
    ~SourcePosition() = default;

    static SourcePosition deserialize(const QJsonValue& jsonSource);

    float azimuth;
    float elevation;
};
}    // namespace Model

#endif    // SOURCE_POSITION_H
