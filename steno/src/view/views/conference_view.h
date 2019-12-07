#ifndef CONFERENCEVIEW_H
#define CONFERENCEVIEW_H

#include "model/media/media.h"
#include "view/views/abstract_view.h"

#include <memory>

#include <QCamera>
#include <QCameraInfo>

namespace Ui
{
class ConferenceView;
}

namespace View
{
class ConferenceView : public AbstractView
{
   public:
    explicit ConferenceView(std::shared_ptr<Model::Media> media, QWidget* parent = nullptr);

   protected:
    void showEvent(QShowEvent* event);
    void hideEvent(QHideEvent* event);

   private:
    Ui::ConferenceView* m_ui;
    std::shared_ptr<Model::Media> m_media;
};

}    // namespace View

#endif    // CONFERENCEVIEW_H
