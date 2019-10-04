#include "playback_view.h"
#include "ui_playback_view.h"

namespace View
{

PlaybackView::PlaybackView(QWidget *parent)
    : AbstractView("Playback", parent)
    , ui(new Ui::PlaybackView)
{
    ui->setupUi(this);
}

PlaybackView::~PlaybackView()
{
    delete ui;
}

} // View
