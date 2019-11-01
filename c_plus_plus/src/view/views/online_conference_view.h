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

namespace Model
{
enum class StreamStatus;
}

namespace View
{
class OnlineConferenceView : public AbstractView
{
    Q_OBJECT

   public:
    explicit OnlineConferenceView(std::shared_ptr<Model::IStream> stream, QWidget *parent = nullptr);
    virtual ~OnlineConferenceView();

   signals:
    void streamCrashed(QPrivateSignal);

   private slots:
    void onStoppedStateEntered();
    void onStartedStateEntered();

   private:
    void onStreamStatusChanged();

    Ui::OnlineConferenceView *m_ui;
    QStateMachine *m_stateMachine;
    QState *m_stopped;
    QState *m_started;
    QState *m_currentState;
    std::shared_ptr<Model::IStream> m_stream;
};

}    // namespace View

#endif    // ONLINE_CONFERENCE_VIEW_H
