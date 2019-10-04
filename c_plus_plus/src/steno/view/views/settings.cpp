#include "settings.h"
#include "ui_settings.h"

namespace View
{

Settings::Settings(QWidget *parent)
    : AbstractView("Settings", parent)
    , ui(new Ui::Settings)
{
    ui->setupUi(this);
}

Settings::~Settings()
{
    delete ui;
}

} // View
