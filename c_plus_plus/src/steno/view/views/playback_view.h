#ifndef PLAYBACK_VIEW_H
#define PLAYBACK_VIEW_H

#include "view/views/abstract_view.h"

namespace Ui { class PlaybackView; }

namespace View
{

class PlaybackView : public AbstractView
{
    public:
        explicit PlaybackView(QWidget *parent = nullptr);
        virtual ~PlaybackView();

    private:
        Ui::PlaybackView *ui;
};

} // View

#endif // PLAYBACK_VIEW_H
