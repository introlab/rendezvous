#ifndef I_RECORDER_H
#define I_RECORDER_H

#include <memory>

#include <QCameraViewfinder>
#include <QWidget>

namespace Model
{
class IRecorder : public QWidget
{
    Q_OBJECT

   public:
    enum State
    {
        Started,
        Stopped
    };

    IRecorder(QWidget* parent = nullptr)
        : QWidget(parent)
    {
    }
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void setCameraViewFinder(std::shared_ptr<QCameraViewfinder> cameraViewFinder) = 0;

    virtual IRecorder::State state() const = 0;

   signals:
    void stateChanged(const IRecorder::State& state);
};

}    // namespace Model

#endif    // I_RECORDER_H
