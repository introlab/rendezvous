#ifndef ODAS_POSITION_SOURCE_H
#define ODAS_POSITION_SOURCE_H

#include <memory>

#include <QThread>

#include "model/stream/audio/i_position_source.h"

class QMutex;

namespace Model
{
class OdasPositionSource : public QThread, public IPositionSource
{
   public:
    OdasPositionSource(int port, int positionBufferSize);
    ~OdasPositionSource() override;

    void open() override;
    void close() override;
    std::vector<SourcePosition> getPositions() override;

   private:
    void run() override;
    void updatePositions(std::vector<SourcePosition>& sourcePositions);

    int m_port;
    std::unique_ptr<QMutex> m_mutex;
    std::vector<SourcePosition> m_sourcePositions;
    std::shared_ptr<char> m_readBuffer;
};

}    // namespace Model

#endif    // ODAS_POSITION_SOURCE_H
