#include "odas_client.h"
#include "model/app_constants.h"

#include <QDebug>
#include <QFile>
#include <QProcess>

namespace Model
{
void OdasClient::run()
{
    qInfo() << "Odaslive thread started";
    m_state = OdasClientState::RUNNING;
    notify();

    const QString& program = Model::ODAS_LIBRARY;
    QStringList arguments;
    arguments << "-c" << Model::MICROPHONE_CONFIGURATION;

    if (!QFile::exists(program) || !QFile::exists(Model::MICROPHONE_CONFIGURATION))
    {
        qCritical() << "Cannot find odaslive and/or microphones config file: " << program << " "
                    << Model::MICROPHONE_CONFIGURATION;
        m_state = OdasClientState::CRASHED;
        notify();
        qInfo() << "Odaslive thread finished";
        return;
    }

    QProcess process;
    process.start(program, arguments);

    bool isProcessFinished = false;
    QByteArray output;

    while (!isAbortRequested())
    {
        isProcessFinished = process.waitForFinished(m_waitTime);
        if (isProcessFinished)
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
            qInfo() << "Odaslive thread finished";
            return;
        }
    }

    closeProcess(process);
    m_state = OdasClientState::STOPPED;
    notify();
    qInfo() << "Odaslive thread finished";
}

void OdasClient::closeProcess(QProcess& process)
{
    // Ensure the process clean close.
    process.close();
    process.waitForFinished(m_joinTime);
    if (process.isOpen())
    {
        qWarning() << "Odaslive was killed.";
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

void OdasClient::detach(IObserver* observer)
{
    for (int index = 0; static_cast<size_t>(index) < m_subscribers.size(); ++index)
    {
        if (m_subscribers.at(static_cast<size_t>(index)) == observer)
        {
            m_subscribers.erase(m_subscribers.begin() + index);
        }
    }
}

void OdasClient::notify()
{
    for (auto observer : m_subscribers)
    {
        observer->updateObserver();
    }
}

OdasClientState OdasClient::getState()
{
    return m_state;
}
}    // namespace Model
