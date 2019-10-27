#include <QCoreApplication>
#include <QDebug>
#include <QThread>
#include <memory>
#include "model/audio/i_audio_source.h"
#include "model/audio/i_position_source.h"
#include "model/audio/odas/odas_audio_source.h"
#include "model/audio/odas/odas_client.h"
#include "model/audio/odas/odas_position_source.h"
#include "model/audio/source_position.h"
#include "view/mainwindow.h"
#define AUDIO_BUFFER_SIZE 4096

int main(int argc, char* argv[])
{
    QCoreApplication a(argc, argv);

    QThread* thread = QThread::create([&] {
        std::unique_ptr<Model::IPositionSource> odasPositionSource = std::make_unique<Model::OdasPositionSource>(10020);
        if (!odasPositionSource->open()) exit(-1);
        std::unique_ptr<Model::IAudioSource> odasAudioSource = std::make_unique<Model::OdasAudioSource>(10030);
        if (!odasAudioSource->open()) exit(-1);
        FILE* m_file = fopen("allo.raw", "ab");
        uint8_t audioBuffer[ AUDIO_BUFFER_SIZE ];

        std::unique_ptr<Model::OdasClient> odasClient = std::make_unique<Model::OdasClient>();
        odasClient->start();

        while (true)
        {
            QCoreApplication::processEvents();
            int audioBytesRead = odasAudioSource->read(audioBuffer, AUDIO_BUFFER_SIZE);
            if (audioBytesRead > 0)
            {
                fwrite(audioBuffer, sizeof(audioBuffer[ 0 ]), audioBytesRead, m_file);
            }
            std::vector<Model::SourcePosition> positions = odasPositionSource->getPositions();
            for (auto pos : positions)
            {
                qDebug() << "azimuth = " << pos.azimuth << ", elevation = " << pos.elevation;
            }
            QThread::usleep(100);
        }
        fclose(m_file);
        odasClient->stop();
    });
    thread->start();
    return a.exec();
}
