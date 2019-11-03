#ifndef RECORDING_VIEW_H
#define RECORDING_VIEW_H

#include "model/recorder/i_recorder.h"
#include "model/stream/i_stream.h"
#include "view/views/abstract_view.h"

namespace Ui
{
class LocalConferenceView;
}

namespace View
{
class LocalConferenceView : public AbstractView
{
   public:
    explicit LocalConferenceView(std::shared_ptr<Model::IStream> stream, std::shared_ptr<Model::IRecorder> recorder,
                                 QWidget* parent = nullptr);

   private slots:
    void onStartVirtualDevicesButtonClicked();
    void onStartRecorderButtonClicked();
    void onStreamStateChanged(const Model::IStream::State& state);
    void onRecorderStateChanged(const Model::IRecorder::State& state);

   private:
    Ui::LocalConferenceView* m_ui;
    std::shared_ptr<QCameraViewfinder> m_cameraViewfinder;
    std::shared_ptr<Model::IStream> m_stream;
    std::shared_ptr<Model::IRecorder> m_recorder;
};

}    // namespace View

#endif    // RECORDING_VIEW_H
