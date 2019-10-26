#ifndef ODAS_POSITION_SOURCE_H
#define ODAS_POSITION_SOURCE_H

#include <memory>

#include "model/audio/i_position_source.h"
#include "model/network/local_socket_server.h"

constexpr int POSITION_BUFFER_SIZE = 10000;

class QMutex;

namespace Model
{
class OdasPositionSource : public QObject, public IPositionSource
{
    Q_OBJECT

   public:
    OdasPositionSource(quint16 port);
    ~OdasPositionSource() override;

    bool open() override;
    bool close() override;
    std::vector<SourcePosition> getPositions() override;

   private slots:
    void onPositionsReady(int numberOfBytes);

   private:
    void updatePositions(std::vector<SourcePosition>& sourcePositions);

    std::unique_ptr<LocalSocketServer> m_socketServer;
    std::unique_ptr<QMutex> m_mutex;
    std::vector<SourcePosition> m_sourcePositions;
    std::array<char, POSITION_BUFFER_SIZE> m_buffer;
};

}    // namespace Model

#endif    // ODAS_POSITION_SOURCE_H
