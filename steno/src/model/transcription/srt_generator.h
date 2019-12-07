#ifndef SRTGENERATOR_H
#define SRTGENERATOR_H

#include "model/app_config.h"

#include <QObject>

#include <memory>
#include <string>

namespace Model
{
class SrtGenerator : public QObject
{
    Q_OBJECT

   public:
    SrtGenerator(std::shared_ptr<AppConfig> appConfig, QObject* parent);
    void generateSrtFile(const QString& filename, QJsonArray transcriptionWords);

   private:
    QString getSrtBlock(const int blockId, double startTime, double endTime, QString text);
    void getWordInfos(const QJsonObject& transcriptionWord, QString& word, double& wordStartTime, double& wordEndTime);

    const int m_maxTimeForSrtBlock = 6;
    const int m_maxCharInSrtLine = 35;
    const std::shared_ptr<AppConfig> m_appConfig;
};

}    // namespace Model
#endif    // SRTGENERATOR_H
