#include "settings_view.h"
#include "ui_settings_view.h"

namespace View
{

SettingsView::SettingsView(QWidget *parent)
    : AbstractView("Settings", parent)
    , ui(new Ui::SettingsView)
{
    ui->setupUi(this);
}

} // View
