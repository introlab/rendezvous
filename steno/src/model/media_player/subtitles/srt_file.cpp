#include "srt_file.h"

#include "model/utils/time.h"

#include <QFile>
#include <QRegularExpression>
#include <QTextStream>

namespace Model
{
/**
 * @brief Parse a srt file and extract the subtitles.
 * @param [IN] path - absolute path to the file to parse.
 * @return Vector of subtitles.
 */
std::vector<SubtitleItem> SrtFile::parse(const QString& path)
{
    std::vector<SubtitleItem> subtitles;

    QFile file(path);
    if (file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream in(&file);
        in.setCodec("UTF-8");

        const QString& content = in.readAll();

        const QRegularExpression patternStr(
            R"((\d+).*?\n(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3}).*?\n([\s\S]*?)\n\n)");

        QRegularExpressionMatchIterator it = patternStr.globalMatch(content);

        SubtitleItem previousSubtitle = SubtitleItem(0, 0, 0, QString());
        while (it.hasNext())
        {
            const QRegularExpressionMatch& m = it.next();
            SubtitleItem currentSubtitle = SubtitleItem(
                m.captured(1).toUInt(), Time::milliseconds(m.captured(2), m.captured(3), m.captured(4), m.captured(5)),
                Time::milliseconds(m.captured(6), m.captured(7), m.captured(8), m.captured(9)), m.captured(10));

            if (!(currentSubtitle.start < previousSubtitle.start || currentSubtitle.end < currentSubtitle.start))
            {
                subtitles.push_back(currentSubtitle);
                previousSubtitle = currentSubtitle;
            }
        }
    }

    return subtitles;
}

}    // namespace Model
