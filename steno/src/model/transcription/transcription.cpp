#include "transcription.h"

#include <memory>

#include <QFile>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QSslCertificate>
#include <QSslConfiguration>
#include <QSslKey>
#include <QUrl>

namespace Model
{
Transcription::Transcription(QObject* parent)
    : QObject(parent)
{
    m_manager = std::make_unique<QNetworkAccessManager>();

    connect(m_manager.get(), &QNetworkAccessManager::finished, [=](QNetworkReply* reply) { emit finished(reply); });
}

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

void Transcription::requestTranscription()
{
    QNetworkRequest request;
    request.setSslConfiguration(m_sslConfig);
    request.setUrl(m_url);
    m_manager->get(request);
}
}    // namespace Model
