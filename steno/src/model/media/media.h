#ifndef MEDIA_H
#define MEDIA_H

#include "model/config/config.h"
#include "model/stream/i_stream.h"

#include <memory>

#include <QCamera>
#include <QCameraViewfinder>
#include <QMediaRecorder>
#include <QObject>

namespace Model
{
class Media : public QObject
{
    Q_OBJECT
   public:
    explicit Media(std::shared_ptr<Config> config, std::shared_ptr<IStream> stream);

    void setViewFinder(QCameraViewfinder* view);

    void startRecorder();
    void stopRecorder();
    QMediaRecorder::State recorderState() const;

   signals:
    void recorderStateChanged(QMediaRecorder::State state);
    void recorderAvailabilityChanged(bool available);

   private slots:
    void onStreamStateChanged(const IStream::State& state);

   private:
    void initCamera();
    void initRecorder();

    std::shared_ptr<BaseConfig> m_appConfig;
    std::shared_ptr<BaseConfig> m_videoConfig;
    std::shared_ptr<IStream> m_stream;
    QScopedPointer<QCamera> m_camera;
    QScopedPointer<QMediaRecorder> m_mediaRecorder;
};

}    // namespace Model
#endif    // MEDIA_
