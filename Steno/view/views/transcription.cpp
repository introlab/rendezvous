#include "transcription.h"
#include "ui_transcription.h"

namespace View
{

Transcription::Transcription(QWidget *parent)
    : AbstractView("Transcription", parent)
    , ui(new Ui::Transcription)
{
    ui->setupUi(this);
}

Transcription::~Transcription()
{
    delete ui;
}

} // View
