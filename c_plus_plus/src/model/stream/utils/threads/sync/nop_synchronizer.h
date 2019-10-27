#ifndef NOP_SYNCHRONIZER_H
#define NOP_SYNCHRONIZER_H

#include "model/stream/utils/threads/sync/i_synchronizer.h"

namespace Model
{
class NopSynchronizer : public ISynchronizer
{
   public:
    void sync() const override{};
};

}    // namespace Model

#endif    //! NOP_SYNCHRONIZER_H
