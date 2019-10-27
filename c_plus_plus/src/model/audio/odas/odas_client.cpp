#include "odas_client.h"
#include "model/settings/settings_constants.h"

#include <QDebug>
#include <QProcess>
#include <QThread>

namespace Model
{
OdasClient::OdasClient() : m_pProcess(std::make_unique<QProcess>())
{
    m_pProcess->moveToThread(this);
    qInfo() << Model::ODAS_LIBRARY;
    qInfo() << Model::MICROPHONE_CONFIGURATION;
}

void OdasClient::run()
{
    m_isRunning = true;

    QString program = Model::ODAS_LIBRARY;
    QStringList arguments;
    arguments << "-c" << Model::MICROPHONE_CONFIGURATION;
    m_pProcess->start(program, arguments);

    while (m_isRunning)
    {
        if (!m_pProcess->isOpen() || m_pProcess->exitStatus() == QProcess::CrashExit)
        {
            qCritical() << "Odaslive stopped working.";
            return;
        }

        qInfo() << "running odas";
        this->msleep(1000);    // ms
    }

    m_pProcess->close();
    m_pProcess->waitForFinished();
    if (m_pProcess->isOpen())
    {
        qWarning() << "Odaslive were killed.";
        m_pProcess->kill();
    }
}
}
