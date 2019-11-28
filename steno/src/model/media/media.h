#ifndef MEDIA_H
#define MEDIA_H

#include "model/config/config.h"

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
    explicit Media(std::shared_ptr<Config> config);

    void setViewFinder(QCameraViewfinder* view);

    void startRecorder();
    void stopRecorder();
    QMediaRecorder::State recorderState() const;
    void unLoadCamera();

   signals:
    void recorderStateChanged(QMediaRecorder::State state);

   private:
    void initCamera();
    void initRecorder();

    std::shared_ptr<BaseConfig> m_appConfig;
    std::shared_ptr<BaseConfig> m_videoConfig;
    QScopedPointer<QCamera> m_camera;
    QScopedPointer<QMediaRecorder> m_mediaRecorder;
};

}    // namespace Model
#endif    // MEDIA_
