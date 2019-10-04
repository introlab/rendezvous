#ifndef TRANSCRIPTION_H
#define TRANSCRIPTION_H

#include <QWidget>

namespace Ui { class Transcription; }

namespace View
{

class Transcription : public QWidget
{
    Q_OBJECT

    public:
        explicit Transcription(QWidget *parent = nullptr);
        virtual ~Transcription();

    private:
        Ui::Transcription *ui;
};

} // View

#endif // TRANSCRIPTION_H
