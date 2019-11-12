#ifndef CONFERENCEVIEW_H
#define CONFERENCEVIEW_H

#include "model/recorder/i_recorder.h"
#include "view/views/abstract_view.h"

#include <memory>

#include <QCameraViewfinder>

namespace Ui
{
class ConferenceView;
}

namespace View
{

class ConferenceView : public AbstractView
{
   public:
    explicit ConferenceView(std::shared_ptr<Model::IRecorder> recorder, QWidget* parent = nullptr);
   private:
    Ui::ConferenceView* m_ui;
    std::shared_ptr<QCameraViewfinder> m_cameraViewfinder;
};

}

#endif // CONFERENCEVIEW_H
