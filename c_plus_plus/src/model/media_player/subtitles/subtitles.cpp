#include "subtitles.h"

#include "srt_file.h"

#include <cmath>

namespace Model
{
Subtitles::Subtitles(QObject *parent)
    : QObject(parent)
    , m_subtitles({})
    , m_lastSubtitleIndex(0)
    , m_currentTime(0)
    , m_lastSubtitleSignaled(QString())
    , m_timer(parent)
{
    m_timer.setTimerType(Qt::PreciseTimer);
    m_timer.setInterval(m_timerInterval);

    connect(&m_timer, &QTimer::timeout, [=] { onTimerTimeout(); });
}

void Subtitles::open(const QString &srtFilePath)
{
    m_subtitles = SrtFile::parse(srtFilePath);
}

void Subtitles::play()
{
    if (!m_subtitles.empty())
    {
        m_timer.start();
        subtitleChanged(currentSubtitle(m_currentTime, true));
    }
}

void Subtitles::pause()
{
    if (!m_subtitles.empty())
    {
        m_timer.stop();
    }
}

void Subtitles::stop()
{
    if (!m_subtitles.empty())
    {
        m_timer.stop();
        reset();
    }
}

void Subtitles::setCurrentTime(qint64 time)
{
    if (!m_subtitles.empty())
    {
        if (m_timer.isActive())
        {
            m_timer.start();
        }

        m_currentTime = time;
        subtitleChanged(currentSubtitle(m_currentTime, true));
    }
}

void Subtitles::onTimerTimeout()
{
    m_currentTime += m_timerInterval;
    QString newSubtitle = currentSubtitle(m_currentTime, false);
    if (newSubtitle != m_lastSubtitleSignaled)
    {
        subtitleChanged(newSubtitle);
        m_lastSubtitleSignaled = newSubtitle;
    }
}

void Subtitles::reset()
{
    m_subtitles.clear();
    m_currentTime = 0;
    m_lastSubtitleIndex = 0;
    m_lastSubtitleSignaled = "";
    subtitleChanged(QString());
}

QString Subtitles::currentSubtitle(qint64 time, bool manual)
{
    QString subtitle = "";

    if (m_subtitles.size() != 0 && time < m_subtitles.back().end)
    {
        if (m_lastSubtitleIndex != 0 && !manual)
        {
            //  Linear search for automatic next
            for (quint64 i = m_lastSubtitleIndex, len = m_subtitles.size(); i < len; i++)
            {
                SubtitleItem item = m_subtitles[i];
                if (time >= item.start && time <= item.end)
                {
                    m_lastSubtitleIndex = i;
                    subtitle = item.text;
                    break;
                }
            }
        }
        else
        {
            // Binary Search for initialization or manual changes
            quint64 lo = 0, hi = m_subtitles.size() - 1;
            while (lo < hi)
            {
                quint64 mid = lo + (hi - lo) / 2;
                SubtitleItem item = m_subtitles[mid];
                if (time >= item.start && time <= item.end)
                {
                    lo = mid;
                    break;
                }
                else if (time > item.end)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid;
                }
            }

            if (time >= m_subtitles[lo].start && time <= m_subtitles[lo].end)
            {
                m_lastSubtitleIndex = lo;
                subtitle = m_subtitles.at(lo).text;
            }
        }
    }

    return subtitle;
}

}    // namespace Model
