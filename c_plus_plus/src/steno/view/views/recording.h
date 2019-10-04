#ifndef RECORDING_H
#define RECORDING_H

#include "view/views/abstract_view.h"

namespace Ui { class Recording; }

namespace View
{

class Recording : public AbstractView
{
    public:
        explicit Recording(QWidget *parent = nullptr);
        virtual ~Recording();

    private:
        Ui::Recording *ui;
};

} // View

#endif // RECORDING_H
