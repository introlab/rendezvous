#ifndef ONLINE_CONFERENCE_VIEW_H
#define ONLINE_CONFERENCE_VIEW_H

#include "view/views/abstract_view.h"

class QStateMachine;
class QState;

namespace Ui { class OnlineConferenceView; }

namespace View
{

class OnlineConferenceView : public AbstractView
{
public:
    explicit OnlineConferenceView(QWidget *parent = nullptr);

public slots:
    void onStoppedStateEntered();
    void onStartedStateEntered();

private:
    Ui::OnlineConferenceView *m_ui;
    QStateMachine *m_stateMachine;
    QState *m_stopped;
    QState *m_started;
};

} // View

#endif // ONLINE_CONFERENCE_VIEW_H
