#ifndef TRANSCRIPTION_H
#define TRANSCRIPTION_H

#include "view/views/abstract_view.h"

namespace Ui { class Transcription; }

namespace View
{

class Transcription : public AbstractView
{
    public:
        explicit Transcription(QWidget *parent = nullptr);
        virtual ~Transcription();

    private:
        Ui::Transcription *ui;
};

} // View

#endif // TRANSCRIPTION_H
