#ifndef SETTINGS_H
#define SETTINGS_H

#include "view/views/abstract_view.h"

namespace Ui { class Settings; }

namespace View
{

class Settings : public AbstractView
{
    public:
        explicit Settings(QWidget *parent = nullptr);
        virtual ~Settings();

    private:
        Ui::Settings *ui;
};

} // View

#endif // SETTINGS_H
