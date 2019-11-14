#ifndef TRANSCRIPTION_H
#define TRANSCRIPTION_H

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
    explicit Transcription(QObject* parent = nullptr);

    bool configureRequest();
    bool requestTranscription(QString audioFilePath);

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
                return "OggOpus";
            case Encoding::SPEEX_WITH_HEADER_BYTE:
                return "Speex Wtih Header Byte";
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
                return "FR CA";
            case Language::EN_CA:
                return "EN CA";
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
                return "Default";
            case Model::COMMAND_AND_SEARCH:
                return "Command and search";
            case Model::PHONE_CALL:
                return "Phone call";
            case Model::VIDEO:
                return "Video";
        }
    }

   signals:
    void finished(QNetworkReply* reply);

   private:
    QSslConfiguration m_sslConfig;
    std::unique_ptr<QNetworkAccessManager> m_manager;
    QUrl m_url = QUrl("https://localhost:3000/");
};
}    // namespace Model

#endif    // TRANSCRIPTION_H