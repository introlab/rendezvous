#ifndef I_STREAM_H
#define I_STREAM_H

namespace Model
{
class IStream
{
   public:
    virtual ~IStream() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
};

}    // namespace Model

#endif    //! I_STREAM_H