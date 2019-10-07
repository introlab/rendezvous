#ifndef SETTINGS_VIEW_H
#define SETTINGS_VIEW_H

#include "view/views/abstract_view.h"

namespace Ui { class SettingsView; }

namespace View
{

class SettingsView : public AbstractView
{
    public:
        explicit SettingsView(QWidget *parent = nullptr);
        virtual ~SettingsView();

    private:
        Ui::SettingsView *ui;
};

} // View

#endif // SETTINGS_VIEW_H
