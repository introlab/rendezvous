#include "playback.h"
#include "ui_playback.h"

namespace View
{

Playback::Playback(QWidget *parent)
    : AbstractView("Playback", parent)
    , ui(new Ui::Playback)
{
    ui->setupUi(this);
}

Playback::~Playback()
{
    delete ui;
}

} // View
