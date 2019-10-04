#ifndef PLAYBACK_H
#define PLAYBACK_H

#include <QWidget>

namespace Ui { class Playback; }

namespace View
{

class Playback : public QWidget
{
    Q_OBJECT

    public:
        explicit Playback(QWidget *parent = nullptr);
        virtual ~Playback();

    private:
        Ui::Playback *ui;
};

} // View

#endif // PLAYBACK_H
