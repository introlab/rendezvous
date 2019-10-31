#include "odas_client.h"
#include "model/settings/settings_constants.h"

#include <QDebug>
#include <QFile>
#include <QProcess>
#include <QThread>

namespace Model
{
void OdasClient::run()
{
    m_state = OdasClientState::RUNNING;
    notify();

    const QString& program = Model::ODAS_LIBRARY;
    QStringList arguments;
    arguments << "-c" << Model::MICROPHONE_CONFIGURATION;

    if (!QFile::exists(program) || !QFile::exists(Model::MICROPHONE_CONFIGURATION))
    {
        stop();
    }

    QProcess process;
    process.start(program, arguments);

    bool returnValue = false;
    QByteArray output;
    while (!isAbortRequested())
    {
        returnValue = process.waitForFinished(m_waitTime);
        if (returnValue)
        {
            // Handle the process' crash.
            qCritical() << "Odaslive stopped working.";
            output.append(process.readAll());

            if (!output.isEmpty())
            {
                qWarning() << "Odas output:" << output;
            }

            closeProcess(process);
            m_state = OdasClientState::CRASHED;
            notify();
            break;
        }
    }

    closeProcess(process);
    m_state = OdasClientState::STOPPED;
    notify();
}

void OdasClient::closeProcess(QProcess& process)
{
    // Ensure the process clean close.
    process.close();
    process.waitForFinished(m_joinTime);
    if (process.isOpen())
    {
        qWarning() << "Odaslive were killed.";
        process.kill();
    }
}

void OdasClient::attach(IObserver* observer)
{
    if (observer != nullptr)
    {
        m_subscribers.push_back(observer);
    }
}

void OdasClient::notify()
{
    for (auto observer : m_subscribers)
    {
        if (observer != nullptr)
        {
            observer->updateObserver();
        }
    }
}

OdasClientState OdasClient::getState()
{
    return m_state;
}
}
