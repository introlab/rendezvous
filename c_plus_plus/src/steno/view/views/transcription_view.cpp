#include "transcription_view.h"
#include "ui_transcription_view.h"

namespace View
{

TranscriptionView::TranscriptionView(QWidget *parent)
    : AbstractView("Transcription", parent)
    , ui(new Ui::TranscriptionView)
{
    ui->setupUi(this);
}

} // View
