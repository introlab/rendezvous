#ifndef ONLINE_CONFERENCE_VIEW_H
#define ONLINE_CONFERENCE_VIEW_H

#include <memory>

#include "model/stream/i_stream.h"
#include "view/views/abstract_view.h"

namespace Ui
{
class OnlineConferenceView;
}

namespace View
{
class OnlineConferenceView : public AbstractView
{
    Q_OBJECT

   public:
    explicit OnlineConferenceView(std::shared_ptr<Model::IStream> stream, QWidget *parent = nullptr);
    virtual ~OnlineConferenceView();

   private slots:
    void onStartButtonClicked();
    void onStreamStateChanged(const Model::IStream::State &state);

   private:
    Ui::OnlineConferenceView *m_ui;
    std::shared_ptr<Model::IStream> m_stream;
};

}    // namespace View

#endif    // ONLINE_CONFERENCE_VIEW_H
