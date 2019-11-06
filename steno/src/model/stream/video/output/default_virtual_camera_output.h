#ifndef DEFAULT_VIRTUAL_CAMERA_OUTPUT_H
#define DEFAULT_VIRTUAL_CAMERA_OUTPUT_H

#include <QString>

namespace Model
{

class DefaultVirtualCameraOutput
{
public:
    static void writeDefaultImage(const QString &deviceName);
};

} // Model
#endif // DEFAULT_VIRTUAL_CAMERA_OUTPUT_H
