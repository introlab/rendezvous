#ifndef CONFERENCE_H
#define CONFERENCE_H

#include <QWidget>

namespace Ui { class Conference; }

namespace View
{

class Conference : public QWidget
{
    Q_OBJECT

    public:
        explicit Conference(QWidget *parent = nullptr);
        virtual ~Conference();

    private:
        Ui::Conference *ui;
};

} // View

#endif // CONFERENCE_H
