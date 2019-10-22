#ifndef I_SYNCHRONIZER_H
#define I_SYNCHRONIZER_H

class ISynchronizer
{
public:

    virtual ~ISynchronizer() = default;
    virtual void sync() const = 0;

};

#endif //!I_SYNCHRONIZER_H