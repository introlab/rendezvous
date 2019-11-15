#include "conference_view.h"
#include "ui_conference_view.h"

namespace View
{
ConferenceView::ConferenceView(std::shared_ptr<Model::Media> media, QWidget* parent)
    : AbstractView("Conference", parent)
    , m_ui(new Ui::ConferenceView)
{
    m_ui->setupUi(this);
    media->setViewFinder(m_ui->cameraViewFinder);
}

}    // namespace View
