#include "transcription.h"
#include "model/app_config.h"
#include "model/config/config.h"
#include "transcription_config.h"

#include <memory>

#include <QFile>
#include <QHttpMultiPart>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>
#include <QSslCertificate>
#include <QSslConfiguration>
#include <QSslKey>
#include <QUrl>
#include <QUrlQuery>

namespace Model
{
Transcription::Transcription(std::shared_ptr<Config> config, QObject* parent)
    : QObject(parent)
    , m_config(config)
{
    m_manager = std::make_unique<QNetworkAccessManager>();

    connect(m_manager.get(), &QNetworkAccessManager::finished, [=](QNetworkReply* reply) {
        if (reply->error() != QNetworkReply::NoError)
        {
            emit finished(false, reply->errorString());
        }

        QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
        postTranscription(doc);
        emit finished(true, "transcription done");
    });
}

/**
 * Transcribe a video file.
 * @param [IN] videoFilePath - video filepath to transcrive the audio.
 * @return true/false if success
 */
bool Transcription::transcribe(const QString& videoFilePath)
{
    bool isOK = prepareTranscription(videoFilePath);
    if (!isOK) return isOK;

    isOK = requestTranscription();
    return isOK;
}

/**
 * @brief Prepare the ssl configuration and generate the audio file from the video file in input.
 * @param [IN] videoFilePath - video filepath to transcribe the audio.
 * @return true/false if success
 */
bool Transcription::prepareTranscription(const QString& videoFilePath)
{
    const auto appConfig = m_config->appConfig();
    const QString& outputFolder = appConfig->value(AppConfig::OUTPUT_FOLDER).toString();
    m_tempWavFilePath = outputFolder + "/audio.wav";
    bool isOK = extractAudioFromVideo(videoFilePath);
    if (!isOK) return false;

    isOK = configureRequest();
    return isOK;
}

/**
 * @brief Transcription::extractAudioFromVideo
 * @param videoFilePath - file to extract the audio
 * @return true/false if success
 */
bool Transcription::extractAudioFromVideo(const QString& videoFilePath)
{
    if (!QFile::exists(videoFilePath))
    {
        qCritical() << "Cannot find video file:" << videoFilePath;
        return false;
    }

    QProcess process;
    const QString command = "ffmpeg -i " + videoFilePath + " -f wav " + m_tempWavFilePath;
    process.start(command);

    process.waitForFinished();
    return process.exitCode() == 0;
}

/**
 * @brief Configure the ssl parameters.
 * @return true/false if the configuration worked
 */
bool Transcription::configureRequest()
{
    m_sslConfig = QSslConfiguration::defaultConfiguration();
    m_sslConfig.setProtocol(QSsl::TlsV1_2);

    QFile certFile("/home/morel/development/rendezvous/transcription_api/ssl/client.crt");
    if (!certFile.exists()) return false;

    certFile.open(QFile::ReadOnly);
    QSslCertificate certificate(&certFile);
    certFile.close();

    QFile keyFile("/home/morel/development/rendezvous/transcription_api/ssl/client.key");
    if (!keyFile.exists()) return false;

    keyFile.open(QFile::ReadOnly);
    QSslKey key(&keyFile, QSsl::Rsa, QSsl::Pem, QSsl::PrivateKey, "password");
    keyFile.close();

    m_sslConfig.setPrivateKey(key);
    m_sslConfig.setLocalCertificate(certificate);

    // TODO: remove this when deployed
    m_sslConfig.setPeerVerifyMode(QSslSocket::VerifyNone);
    return true;
}

/**
 * @brief Ask the transcription server for a transcript of a wav file in input.
 * @return true/false if success
 */
bool Transcription::requestTranscription()
{
    // body construction (put the audio file in the request)
    QFile* audioFile = new QFile(m_tempWavFilePath);
    if (!audioFile->exists()) return false;
    audioFile->open(QIODevice::ReadOnly);

    QHttpMultiPart* multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);
    QHttpPart textPart;
    textPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"name\""));
    textPart.setBody("audio");    // audio is the name given to the uploaded file in the server.
    multiPart->append(textPart);

    QHttpPart audioPart;
    audioPart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("audio/wav"));
    audioPart.setHeader(QNetworkRequest::ContentDispositionHeader,
                        QVariant("form-data; name=\"audio\"; filename=\"audio.wav\""));
    audioPart.setBodyDevice(audioFile);
    audioFile->setParent(multiPart);    // we can't delete the audioFile pointer, so we set the multiPart has the
                                        // parent. This way Qt will destroy audioFile pointer with the MultiPart.
    multiPart->append(audioPart);

    // Query parameters construction.
    QUrlQuery query;
    query.addQueryItem("encoding", encodingName(Encoding::LINEAR16));
    query.addQueryItem("enhanced", "true");

    const auto transcriptionConfig = m_config->transcriptionConfig();
    const Language lang = static_cast<Language>(transcriptionConfig->value(TranscriptionConfig::LANGUAGE).toInt());
    query.addQueryItem("language", languageName(lang));

    query.addQueryItem("sampleRate", "48000");
    query.addQueryItem("audioChannels", "2");
    query.addQueryItem("model", modelName(Model::DEFAULT));
    // TODO: Make a UID generator that each Jetson will use to acquire a unique ID and use it as bucketID.
    query.addQueryItem("bucketID", "steno1");

    m_url.setQuery(query);

    QNetworkRequest request;
    request.setSslConfiguration(m_sslConfig);
    request.setUrl(m_url);

    QNetworkReply* reply = m_manager->post(request, multiPart);
    multiPart->setParent(reply);    // We set the parent of the multiPart to the reply. This way when the reply is done,
                                    // Qt will delete the multiPart and the audioFile pointers.

    return true;
}

/**
 * @brief Delete the temporary audio file generated and generate the srt file.
 * @param response - json document of the transcription request.
 * @return true/false if success
 */
bool Transcription::postTranscription(QJsonDocument response)
{
    qDebug() << response;
    bool isOK = deleteFile();
    if (!isOK) return false;
    // TODO: ask srt file generation.
    return true;
}

/**
 * @brief Delete the temp audio file from the system.
 * @return true/false if success
 */
bool Transcription::deleteFile()
{
    const QString program = "rm";
    QStringList arguments;
    arguments << m_tempWavFilePath;

    if (!QFile::exists(m_tempWavFilePath))
    {
        qCritical() << "Cannot find file to delete:" << m_tempWavFilePath;
        return false;
    }

    QProcess process;
    process.start(program, arguments);

    process.waitForFinished();
    return process.exitCode() == 0;
}
}    // namespace Model
