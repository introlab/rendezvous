#include "odas_client.h"
#include "model/settings/settings_constants.h"

#include <QDebug>
#include <QProcess>
#include <QThread>

namespace Model
{
OdasClient::OdasClient()
    : m_pProcess(std::make_unique<QProcess>())
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

    bool returnValue = false;
    QByteArray output;
    while (m_isRunning)
    {
        returnValue = m_pProcess->waitForFinished(1000);
        if (returnValue)
        {
            qCritical() << "Odaslive stopped working.";
            output.append(m_pProcess->readAll());
            if (!output.isEmpty())
            {
                qWarning() << "Odas output:" << output;
            }
            return;
        }
    }

    m_pProcess->close();
    m_pProcess->waitForFinished(500);
    if (m_pProcess->isOpen())
    {
        qWarning() << "Odaslive were killed.";
        m_pProcess->kill();
    }
}
}
