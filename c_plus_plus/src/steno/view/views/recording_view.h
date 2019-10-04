#ifndef RECORDING_VIEW_H
#define RECORDING_VIEW_H

#include "view/views/abstract_view.h"

namespace Ui { class RecordingView; }

namespace View
{

class RecordingView : public AbstractView
{
    public:
        explicit RecordingView(QWidget *parent = nullptr);
        virtual ~RecordingView();

    private:
        Ui::RecordingView *ui;
};

} // View

#endif // RECORDING_VIEW_H
