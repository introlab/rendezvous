#include "transcription.h"
#include "ui_transcription.h"

namespace View
{

Transcription::Transcription(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Transcription)
{
    ui->setupUi(this);
}

Transcription::~Transcription()
{
    delete ui;
}

} // View
