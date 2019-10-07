#ifndef TRANSCRIPTION_VIEW_H
#define TRANSCRIPTION_VIEW_H

#include "view/views/abstract_view.h"

namespace Ui { class TranscriptionView; }

namespace View
{

class TranscriptionView : public AbstractView
{
    public:
        explicit TranscriptionView(QWidget *parent = nullptr);
        virtual ~TranscriptionView();

    private:
        Ui::TranscriptionView *ui;
};

} // View

#endif // TRANSCRIPTION_VIEW_H
