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
    void requestTranscription();

   signals:
    void finished(QNetworkReply* reply);

   private:
    QSslConfiguration m_sslConfig;
    std::unique_ptr<QNetworkAccessManager> m_manager;
    const QUrl m_url = QUrl("https://localhost:3000/");
};
}    // namespace Model

#endif    // TRANSCRIPTION_H
