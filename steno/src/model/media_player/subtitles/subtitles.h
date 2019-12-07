#ifndef SUBTITLES_H
#define SUBTITLES_H

#include "subtitle_item.h"

#include <vector>

#include <QObject>
#include <QString>
#include <QTimer>

namespace Model
{
class SubtitleItem;

class Subtitles : public QObject
{
    Q_OBJECT

   public:
    Subtitles(QObject *parent = nullptr);

    void open(const QString &srtFilePath);
    void play();
    void pause();
    void stop();
    void clear();
    void setCurrentTime(qint64 time);

   signals:
    void subtitleChanged(const QString &subtitle);

   private slots:
    void onTimerTimeout();

   private:
    void reset();
    QString currentSubtitle(qint64 time, bool manual);
    void linearSearch(qint64 time, QString &subtitle);
    void binarySearch(qint64 time, QString &subtitle);

    std::vector<SubtitleItem> m_subtitles;
    quint64 m_lastSubtitleIndex;
    qint64 m_currentTime;
    QString m_lastSubtitleSignaled;
    QTimer m_timer;

    const int m_timerInterval = 200;
};

}    // namespace Model

#endif    // SUBTITLES_H
