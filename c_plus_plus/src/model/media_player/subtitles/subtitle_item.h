#ifndef SUBTITLE_ITEM_H
#define SUBTITLE_ITEM_H

#include <QString>

namespace Model
{
struct SubtitleItem
{
    SubtitleItem(quint64 id, qint64 start, qint64 end, QString text)
    {
        this->id = id;
        this->start = start;
        this->end = end;
        this->text = text;
    }
    ~SubtitleItem() = default;

    bool operator==(const SubtitleItem &rhs) const
    {
        return id == rhs.id && start == rhs.start && end == rhs.end && text == rhs.text;
    }

    quint64 id;
    qint64 start, end;
    QString text;
};
}    // namespace Model

#endif    // SUBTITLE_ITEM_H
