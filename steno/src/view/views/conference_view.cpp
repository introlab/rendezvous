#include "conference_view.h"
#include "ui_conference_view.h"

namespace View
{
ConferenceView::ConferenceView(std::shared_ptr<Model::Media> media, QWidget* parent)
    : AbstractView("Conference", parent)
    , m_ui(new Ui::ConferenceView)
    , m_media(media)
{
    m_ui->setupUi(this);
    m_media->setViewFinder(m_ui->cameraViewFinder);
}

void ConferenceView::showEvent(QShowEvent* /*event*/)
{
    m_media->setViewFinder(m_ui->cameraViewFinder);
}

void ConferenceView::hideEvent(QHideEvent* /*event*/)
{
    m_media->unLoadCamera();
}
}    // namespace View
