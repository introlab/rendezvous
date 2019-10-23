#ifndef SOURCE_POSITION_H
#define SOURCE_POSITION_H

#include <QJsonValue>


namespace Model
{

struct SourcePosition
{
    SourcePosition(double azimuth, double elevation);
    ~SourcePosition();

    static SourcePosition deserialize(const QJsonValue jsonValue);

    double azimuth;
    double elevation;
};

}

#endif // SOURCE_POSITION_H
