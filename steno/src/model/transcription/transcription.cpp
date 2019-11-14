#include "transcription.h"
#include "transcription_config.h"

#include <memory>

#include <QFile>
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
Transcription::Transcription(QObject* parent)
    : QObject(parent)
{
    m_manager = std::make_unique<QNetworkAccessManager>();

    connect(m_manager.get(), &QNetworkAccessManager::finished, [=](QNetworkReply* reply) { emit finished(reply); });
}

/**
 * @brief Transcription::configureRequest configure the ssl parameters.
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
 * @brief Transcription::requestTranscription send the transcription request and the signal finished is emit when the
 * response is received.
 * @return true/false if the request was send correctly.
 */
bool Transcription::requestTranscription(QString audioFilePath)
{
    QFile audioFile(audioFilePath);
    if (!audioFile.exists()) return false;

    QUrlQuery query;
    query.addQueryItem("encoding", encodingName(Encoding::LINEAR16));
    query.addQueryItem("enhanced", "true");
    // TODO: get the language from the config.
    query.addQueryItem("language", "");
    query.addQueryItem("sampleRate", "44100");
    query.addQueryItem("audioChannels", "2");
    query.addQueryItem("model", modelName(Model::DEFAULT));
    query.addQueryItem("storage", "true");
    // TODO: Make a UID generator that each Jetson will use to acquire a unique ID and use it as bucketID.
    query.addQueryItem("bucketID", "Steno1");

    m_url.setQuery(query);

    QNetworkRequest request;
    request.setSslConfiguration(m_sslConfig);
    request.setUrl(m_url);
    m_manager->get(request);
    return true;
}
}    // namespace Model
