#ifndef ONLINE_CONFERENCE_VIEW_H
#define ONLINE_CONFERENCE_VIEW_H

#include <memory>

#include "model/stream/i_stream.h"
#include "view/views/abstract_view.h"

class QStateMachine;
class QState;

namespace Ui
{
class OnlineConferenceView;
}

namespace View
{
class OnlineConferenceView : public AbstractView
{
   public:
    explicit OnlineConferenceView(std::shared_ptr<Model::IStream> stream, QWidget *parent = nullptr);
    virtual ~OnlineConferenceView();

   private slots:
    void onStoppedStateEntered();
    void onStartedStateEntered();

   private:
    Ui::OnlineConferenceView *m_ui;
    QStateMachine *m_stateMachine;
    QState *m_stopped;
    QState *m_started;
    std::shared_ptr<Model::IStream> m_stream;
};

}    // namespace View

#endif    // ONLINE_CONFERENCE_VIEW_H
