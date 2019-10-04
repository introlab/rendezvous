#ifndef AUDIO_PROCESSING_VIEW_H
#define AUDIO_PROCESSING_VIEW_H

#include <QWidget>

namespace Ui { class AudioProcessing; }

namespace View
{

class AudioProcessing : public QWidget
{
    Q_OBJECT

    public:
        explicit AudioProcessing(QWidget *parent = nullptr);
        virtual ~AudioProcessing();

    private:
        Ui::AudioProcessing *ui;
};

} // View

#endif // AUDIO_PROCESSING_VIEW_H
