#include "recording.h"
#include "ui_recording.h"

namespace View
{

Recording::Recording(QWidget *parent)
    : AbstractView("Recording", parent)
    , ui(new Ui::Recording)
{
    ui->setupUi(this);
}

Recording::~Recording()
{
    delete ui;
}

} // View
