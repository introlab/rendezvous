#include "audio_processing.h"
#include "ui_audio_processing.h"

namespace View
{

AudioProcessing::AudioProcessing(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::AudioProcessing)
{
    ui->setupUi(this);
}

AudioProcessing::~AudioProcessing()
{
    delete ui;
}

} // View
