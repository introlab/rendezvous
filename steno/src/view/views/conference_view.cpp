#include "conference_view.h"
#include "ui_conference_view.h"

namespace View
{

ConferenceView::ConferenceView(std::shared_ptr<Model::IRecorder> recorder, QWidget* parent)
    : AbstractView("Conference", parent)
    , m_ui(new Ui::ConferenceView)
    , m_cameraViewfinder(new QCameraViewfinder(this))
{
    m_ui->setupUi(this);
    m_ui->layout->addWidget(m_cameraViewfinder.get());
    m_cameraViewfinder->show();
    recorder->setCameraViewFinder(m_cameraViewfinder);
}

}
