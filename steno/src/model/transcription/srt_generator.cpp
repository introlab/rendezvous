#include "srt_generator.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonObject>

#include <math.h>

namespace Model
{
SrtGenerator::SrtGenerator(std::shared_ptr<AppConfig> appConfig, QObject* parent)
    : QObject(parent)
    , m_appConfig(appConfig)
{
}

/**
 * @brief Generate a srt file in the application output folder.
 * @param [IN] filename - name of the file to create.
 * @param [IN] transcriptionWords - Json array of each words transcribed.
 */
void SrtGenerator::generateSrtFile(const QString& filename, QJsonArray transcriptionWords)
{
    if (transcriptionWords.isEmpty()) return;

    QString filePath = m_appConfig->value(AppConfig::OUTPUT_FOLDER).toString() + "/" + filename;
    QFile file(filePath);
    file.open(QFile::WriteOnly);

    QString block;
    QString word;
    double lineStartTime;
    double lineEndTime;
    double wordStartTime;
    double wordEndTime;

    getWordInfos(transcriptionWords[0].toObject(), block, lineStartTime, lineEndTime);
    int id = 1;

    for (auto transcriptionWord : transcriptionWords)
    {
        QJsonObject wordObject = transcriptionWord.toObject();

        getWordInfos(wordObject, word, wordStartTime, wordEndTime);
        QString tmpLine = (block + " " + word).trimmed();

        if (tmpLine.length() < (m_maxCharInSrtLine * 2) && (wordEndTime - lineStartTime) < m_maxTimeForSrtBlock)
        {
            block = tmpLine;
            lineEndTime = wordEndTime;
        }
        else
        {
            file.write(getSrtBlock(id, lineStartTime, lineEndTime, block).toUtf8());

            // new block.
            id++;
            lineStartTime = wordStartTime;
            lineEndTime = wordEndTime;
            block = word;
        }
    }

    file.write(getSrtBlock(id, lineStartTime, lineEndTime, block).toUtf8());

    file.close();
}

/**
 * @brief SRT block format:
 *        ID
 *        HH:MM:SS,mmm ---> HH:MM:SS,mmm
 *        Line 1
 *        Line 2
 *        Blank line
 * @param [IN] blockId
 * @param [IN] startTime
 * @param [IN] endTime
 * @param [IN] text
 * @return A string representing a chunk of subtitles.
 */
QString SrtGenerator::getSrtBlock(const int blockId, double startTime, double endTime, QString text)
{
    // ID
    QString block = QString::number(blockId) + "\n";

    // HH:MM:SS,mmm ---> HH:MM:SS,mmm
    block += QString::number(static_cast<int>(startTime / 360)) + QString::number(static_cast<int>(startTime / 60)) +
             QString::number(static_cast<int>(std::fmod(startTime, 60))) +
             QString::number(static_cast<int>(1000 * std::fmod(startTime, 1))) +
             QString::number(static_cast<int>(endTime / 360)) + QString::number(static_cast<int>(endTime / 60)) +
             QString::number(static_cast<int>(std::fmod(endTime, 60))) +
             QString::number(static_cast<int>(1000 * std::fmod(endTime, 1))) + "\n";

    // Might need to split the text in 2 lines.
    if (text.length() <= m_maxCharInSrtLine)
    {
        // line 1
        block += text + "\n";
    }
    else
    {
        QStringList words = text.split(" ");
        int index = 0;
        QString tmpText = words[0];
        words.removeAt(0);

        for (QString word : words)
        {
            QString preview = tmpText + " " + word;
            if (preview.length() > m_maxCharInSrtLine)
            {
                // Saving remaining iteration by getting remaining words.
                QString rest = "\n" + words[index + 1];
                for (int i = index + 2; i < words.length(); ++i)
                {
                    rest += " " + words[i];
                }
                tmpText += " " + rest;
                break;
            }
            else
            {
                tmpText = preview;
            }
        }

        // Line 1 and 2.
        block += tmpText + "\n";
    }

    // Blank line
    block += "\n";
    return block;
}

/**
 * @brief Extract important informations for the srt file from a json object.
 * @param [IN] transcriptionWord - json object that contains the informations.
 * @param [OUT] word - transcribed word.
 * @param [OUT] wordStartTime - start time of that word.
 * @param [OUT] wordEndTime - end time of that word.
 */
void SrtGenerator::getWordInfos(const QJsonObject& transcriptionWord, QString& word, double& wordStartTime,
                                double& wordEndTime)
{
    word = transcriptionWord["word"].toString();
    wordStartTime =
        transcriptionWord["start_time"]["seconds"].toInt() + transcriptionWord["start_time"]["nanos"].toInt() * 1e-9;
    wordEndTime =
        transcriptionWord["end_time"]["seconds"].toInt() + transcriptionWord["end_time"]["nanos"].toInt() * 1e-9;
}

}    // namespace Model
