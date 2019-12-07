#include "srt_generator.h"

#include <QChar>
#include <QFile>
#include <QJsonArray>
#include <QJsonObject>
#include <QString>

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
    double lineStartTime = 0;
    double lineEndTime = 0;
    double wordStartTime = 0;
    double wordEndTime = 0;

    getWordInfos(transcriptionWords[0].toObject(), block, lineStartTime, lineEndTime);

    transcriptionWords.removeAt(0);
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
 *        HH:MM:SS,mmm --> HH:MM:SS,mmm
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

    // HH:MM:SS,mmm --> HH:MM:SS,mmm
    block += QString("%1").arg(static_cast<qint64>(startTime / 360), 2, 10, QChar('0')) + ":" +
             QString("%1").arg(static_cast<qint64>(startTime / 60), 2, 10, QChar('0')) + ":" +
             QString("%1").arg(static_cast<qint64>(std::fmod(startTime, 60)), 2, 10, QChar('0')) + "," +
             QString("%1").arg(static_cast<qint64>(1000 * std::fmod(startTime, 1)), 3, 10, QChar('0')) + " --> " +
             QString("%1").arg(static_cast<qint64>(endTime / 360), 2, 10, QChar('0')) + ":" +
             QString("%1").arg(static_cast<qint64>(endTime / 60), 2, 10, QChar('0')) + ":" +
             QString("%1").arg(static_cast<qint64>(std::fmod(endTime, 60)), 2, 10, QChar('0')) + "," +
             QString("%1").arg(static_cast<qint64>(1000 * std::fmod(endTime, 1)), 3, 10, QChar('0')) + "\n";

    // Might need to split the text in 2 lines.
    if (text.length() <= m_maxCharInSrtLine)
    {
        // line 1
        block += text + "\n";
    }
    else
    {
        QStringList words = text.split(" ");
        QString tmpText = words[0];

        for (int index = 1; index < words.length(); index++)
        {
            QString preview = tmpText + " " + words[index];
            if (preview.length() > m_maxCharInSrtLine)
            {
                // Saving remaining iteration by getting remaining words.
                QString rest = "\n" + words[index];
                for (int i = index + 1; i < words.length(); i++)
                {
                    if (i >= words.length()) break;
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

    wordStartTime = transcriptionWord["startTime"].toObject()["seconds"].toVariant().toUInt() +
                    transcriptionWord["startTime"].toObject()["nanos"].toVariant().toUInt() * 1e-9;
    wordEndTime = transcriptionWord["endTime"].toObject()["seconds"].toVariant().toUInt() +
                  transcriptionWord["endTime"].toObject()["nanos"].toVariant().toUInt() * 1e-9;

    qDebug() << "word:" << word;
    qDebug() << "word start time" << wordStartTime;
    qDebug() << "word end time" << wordEndTime;
}

}    // namespace Model
