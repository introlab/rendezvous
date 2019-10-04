#ifndef PLAYBACK_H
#define PLAYBACK_H

#include "view/views/abstract_view.h"

namespace Ui { class Playback; }

namespace View
{

class Playback : public AbstractView
{
    public:
        explicit Playback(QWidget *parent = nullptr);
        virtual ~Playback();

    private:
        Ui::Playback *ui;
};

} // View

#endif // PLAYBACK_H
