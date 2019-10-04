#ifndef RECORDING_H
#define RECORDING_H

#include <QWidget>

namespace Ui { class Recording; }

namespace View
{

class Recording : public QWidget
{
    Q_OBJECT

    public:
        explicit Recording(QWidget *parent = nullptr);
        virtual ~Recording();

    private:
        Ui::Recording *ui;
};

} // View

#endif // RECORDING_H
