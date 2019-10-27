#ifndef I_POSITION_SOURCE_H
#define I_POSITION_SOURCE_H

#include <vector>

#include "source_position.h"

namespace Model
{
class IPositionSource
{
   public:
    virtual ~IPositionSource() = default;

    virtual bool open() = 0;
    virtual bool close() = 0;
    virtual std::vector<SourcePosition> getPositions() = 0;
};

}    // namespace Model

#endif    // I_POSITION_SOURCE_H
