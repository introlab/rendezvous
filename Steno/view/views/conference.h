#ifndef CONFERENCE_H
#define CONFERENCE_H

#include "view/views/abstract_view.h"

namespace Ui { class Conference; }

namespace View
{

class Conference : public AbstractView
{
    public:
        explicit Conference(QWidget *parent = nullptr);
        virtual ~Conference();

    private:
        Ui::Conference *ui;
};

} // View

#endif // CONFERENCE_H
