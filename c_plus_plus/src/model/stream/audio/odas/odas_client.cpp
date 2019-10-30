#include "odas_client.h"
#include "model/settings/settings_constants.h"

#include <QDebug>
#include <QProcess>
#include <QThread>

namespace Model
{
void OdasClient::run()
{
    m_isRunning = true;

    const QString& program = Model::ODAS_LIBRARY;
    QStringList arguments;
    arguments << "-c" << Model::MICROPHONE_CONFIGURATION;

    QProcess process;
    process.start(program, arguments);

    bool returnValue = false;
    QByteArray output;
    while (m_isRunning)
    {
        returnValue = process.waitForFinished(m_waitTime);
        if (returnValue)
        {
            qCritical() << "Odaslive stopped working.";
            output.append(process.readAll());

            if (!output.isEmpty())
            {
                qWarning() << "Odas output:" << output;
            }
            return;
        }
    }

    // Ensure the process is closed.
    process.close();
    process.waitForFinished(m_joinTime);
    if (process.isOpen())
    {
        qWarning() << "Odaslive were killed.";
        process.kill();
    }
}

void OdasClient::stop()
{
    m_isRunning = false;
    this->msleep(static_cast<ulong>(m_joinTime));
}
}
