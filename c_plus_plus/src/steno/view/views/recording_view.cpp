#include "recording_view.h"
#include "ui_recording_view.h"

namespace View
{

RecordingView::RecordingView(QWidget *parent)
    : AbstractView("Recording", parent)
    , ui(new Ui::RecordingView)
{
    ui->setupUi(this);
}

} // View
