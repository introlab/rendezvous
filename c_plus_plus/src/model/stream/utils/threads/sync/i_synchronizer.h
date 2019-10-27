#ifndef I_SYNCHRONIZER_H
#define I_SYNCHRONIZER_H

namespace Model
{

class ISynchronizer
{
public:

    virtual ~ISynchronizer() = default;
    virtual void sync() const = 0;

};

} // Model

#endif //!I_SYNCHRONIZER_H
