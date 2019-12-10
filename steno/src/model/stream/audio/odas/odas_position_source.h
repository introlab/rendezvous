#ifndef ODAS_POSITION_SOURCE_H
#define ODAS_POSITION_SOURCE_H

#include <memory>
#include <mutex>

#include <QThread>

#include "model/stream/audio/i_position_source.h"

class QMutex;

namespace Model
{
class OdasPositionSource : public QThread, public IPositionSource
{
   public:
    OdasPositionSource(int port);
    ~OdasPositionSource() override;

    void open() override;
    void close() override;
    std::vector<SourcePosition> getPositions() override;

   private:
    void run() override;
    void updatePositions(std::vector<SourcePosition>& sourcePositions);

    int m_port;
    std::vector<SourcePosition> m_sourcePositions;
    std::mutex m_mutex;
};

}    // namespace Model

#endif    // ODAS_POSITION_SOURCE_H
