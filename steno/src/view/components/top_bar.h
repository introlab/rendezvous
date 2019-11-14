#ifndef TOPBAR_H
#define TOPBAR_H

#include "model/media/media.h"
#include "model/stream/i_stream.h"
#include "model/transcription/transcription.h"

#include <memory>

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
           std::shared_ptr<Model::Transcription> transcription, QWidget* parent = nullptr);

   private slots:
    void onStreamStateChanged(const Model::IStream::State& state);
    void onStartButtonClicked();
    void onRecorderStateChanged(const QMediaRecorder::State& state);
    void onRecordButtonClicked();

   private:
    Ui::TopBar* m_ui;
    std::shared_ptr<Model::IStream> m_stream;
    std::shared_ptr<Model::Media> m_media;
    std::shared_ptr<Model::Transcription> m_transcription;
    const QUrl m_rendezvousMeetUrl = QUrl("https://rendezvous-meet.com/");
};

}    // namespace View

#endif    // TOPBAR_H
