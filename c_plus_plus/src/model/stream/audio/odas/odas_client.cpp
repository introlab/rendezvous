#include "odas_client.h"
#include "model/settings/settings_constants.h"

#include <QDebug>
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
            m_state = OdasClientState::STOPPED;
            break;
        }
    }

    // Ensure the process clean close.
    process.close();
    process.waitForFinished(m_joinTime);
    if (process.isOpen())
    {
        qWarning() << "Odaslive were killed.";
        process.kill();
    }
    notify();
    qInfo() << "donezo";
}

void OdasClient::stop()
{
    Thread::stop();
    m_state = OdasClientState::STOPPED;
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
            observer->update();
        }
    }
}

OdasClientState OdasClient::getState()
{
    return m_state;
}
}
