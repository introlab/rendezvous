#include "conference.h"
#include "ui_conference.h"

namespace View
{

Conference::Conference(QWidget *parent)
    : AbstractView("Conference", parent)
    , ui(new Ui::Conference)
{
    ui->setupUi(this);
}

Conference::~Conference()
{
    delete ui;
}

} // View
