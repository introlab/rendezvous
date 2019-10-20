#ifndef NOP_SYNCHRONIZER_H
#define NOP_SYNCHRONIZER_H

#include "ISynchronizer.h"

class NopSynchronizer : public ISynchronizer
{
public:

    void sync() const override {};

};

#endif //!NOP_SYNCHRONIZER_H