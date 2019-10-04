#ifndef CONFERENCE_VIEW_H
#define CONFERENCE_VIEW_H

#include "view/views/abstract_view.h"

namespace Ui { class ConferenceView; }

namespace View
{

class ConferenceView : public AbstractView
{
    public:
        explicit ConferenceView(QWidget *parent = nullptr);
        virtual ~ConferenceView();

    private:
        Ui::ConferenceView *ui;
};

} // View

#endif // CONFERENCE_VIEW_H
