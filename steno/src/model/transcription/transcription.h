#ifndef TRANSCRIPTION_H
#define TRANSCRIPTION_H

#include "model/config/config.h"

#include <memory>

#include <QNetworkAccessManager>
#include <QObject>
#include <QSslConfiguration>
#include <QUrl>

namespace Model
{
class Transcription : public QObject
{
    Q_OBJECT

   public:
    explicit Transcription(std::shared_ptr<Config> config, QObject* parent = nullptr);

    bool transcribe(const QString& videoFilePath);

    enum Encoding
    {
        ENCODING_UNSPECIFIED,
        LINEAR16,
        FLAC,
        MULAW,
        AMR,
        AMR_WB,
        OGG_OPUS,
        SPEEX_WITH_HEADER_BYTE
    };

    static inline const char* encodingName(Encoding encoding)
    {
        switch (encoding)
        {
            case Encoding::ENCODING_UNSPECIFIED:
                return "Encoding Unspecified";
            case Encoding::LINEAR16:
                return "LINEAR16";
            case Encoding::FLAC:
                return "Free Lossless Audio Codec";
            case Encoding::MULAW:
                return "MULAW";
            case Encoding::AMR:
                return "Adaptive Multi-Rate Narrowband";
            case Encoding::AMR_WB:
                return "Adaptive Multi-Rate Wideband";
            case Encoding::OGG_OPUS:
                return "OGG_OPUS";
            case Encoding::SPEEX_WITH_HEADER_BYTE:
                return "SPEEX_WITH_HEADER_BYTE";
        }
    }

    enum Language
    {
        FR_CA,
        EN_CA,
        COUNT
    };

    static inline const char* languageName(Language language)
    {
        switch (language)
        {
            case Language::FR_CA:
                return "fr-CA";
            case Language::EN_CA:
                return "en-CA";
            default:
                return nullptr;
        }
    }

    enum Model
    {
        DEFAULT,
        COMMAND_AND_SEARCH,
        PHONE_CALL,
        VIDEO
    };

    static inline const char* modelName(Model model)
    {
        switch (model)
        {
            case Model::DEFAULT:
                return "default";
            case Model::COMMAND_AND_SEARCH:
                return "command_and_search";
            case Model::PHONE_CALL:
                return "phone_call";
            case Model::VIDEO:
                return "video";
        }
    }

   signals:
    void finished(bool isOK, QString reply);

   private slots:
    void requestFinished(QNetworkReply* reply);

   private:
    bool prepareTranscription(const QString& videoFilePath);
    bool requestTranscription();
    bool postTranscription(QJsonDocument response);
    bool configureRequest();

    bool extractAudioFromVideo(const QString& videoFilePath);
    bool deleteFile();

    QSslConfiguration m_sslConfig;
    std::unique_ptr<QNetworkAccessManager> m_manager;
    QUrl m_url = QUrl("https://rendezvous-meet.com/transcription-api/transcription");
    std::shared_ptr<Config> m_config;
    QString m_tempWavFilePath;
};
}    // namespace Model

#endif    // TRANSCRIPTION_H
