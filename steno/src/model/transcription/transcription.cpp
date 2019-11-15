#include "transcription.h"
#include "model/config/config.h"
#include "transcription_config.h"

#include <memory>

#include <QFile>
#include <QHttpMultiPart>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSslCertificate>
#include <QSslConfiguration>
#include <QSslKey>
#include <QUrl>
#include <QUrlQuery>

namespace Model
{
Transcription::Transcription(std::shared_ptr<Config> config, QObject* parent)
    : QObject(parent)
    , m_config(config->transcriptionConfig())
{
    m_manager = std::make_unique<QNetworkAccessManager>();

    connect(m_manager.get(), &QNetworkAccessManager::finished, [=](QNetworkReply* reply) { emit finished(reply); });
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
 * @brief Send the transcription request and the signal finished is emit when the
 * response is received.
 * @return true/false if the request was send correctly.
 */
bool Transcription::requestTranscription(QString audioFilePath)
{
    // body construction (put the audio file in the request)
    QFile* audioFile = new QFile(audioFilePath);
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
    // TODO: get the language from the config.
    const Language lang = static_cast<Language>(m_config->value(TranscriptionConfig::LANGUAGE).toInt());
    query.addQueryItem("language", languageName(lang));
    query.addQueryItem("sampleRate", "44100");
    query.addQueryItem("audioChannels", "2");
    query.addQueryItem("model", modelName(Model::DEFAULT));
    query.addQueryItem("storage", "true");
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
}    // namespace Model
