#ifndef TOPBAR_H
#define TOPBAR_H

#include "model/config/config.h"
#include "model/media/media.h"
#include "model/stream/i_stream.h"
#include "model/transcription/transcription.h"

#include <memory>

#include <QTimer>
#include <QTime>
#include <QWidget>

namespace Ui
{
class TopBar;
}

namespace View
{
class TopBar : public QWidget
{
    Q_OBJECT

   public:
    TopBar(std::shared_ptr<Model::IStream> stream, std::shared_ptr<Model::Media> media,
           std::shared_ptr<Model::Transcription> transcription, std::shared_ptr<Model::Config> config,
           QWidget* parent = nullptr);

    void stopThreads();

   private slots:
    void onStreamStateChanged(const Model::IStream::State& state);
    void onStartButtonClicked();
    void onRecorderStateChanged(const QMediaRecorder::State& state);
    void onRecordButtonClicked();
    void onTranscriptionFinished(bool isOK, QString reply);
    void onStreamTimerTimeout();

   private:
    bool askTranscription();

    Ui::TopBar* m_ui;
    std::shared_ptr<Model::IStream> m_stream;
    std::shared_ptr<Model::Media> m_media;
    std::shared_ptr<Model::Transcription> m_transcription;
    std::shared_ptr<Model::TranscriptionConfig> m_transcriptionConfig;
    std::shared_ptr<Model::AppConfig> m_applicationConfig;
    const QUrl m_rendezvousMeetUrl = QUrl("https://rendezvous-meet.com/");

    QTimer m_streamTimer;
    QTime m_streamStartTime;
};

}    // namespace View

#endif    // TOPBAR_H
