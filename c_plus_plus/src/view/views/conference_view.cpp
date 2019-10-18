#include "conference_view.h"
#include "ui_conference_view.h"

namespace View
{

ConferenceView::ConferenceView(QWidget *parent)
    : AbstractView("Conference", parent)
    , ui(new Ui::ConferenceView)
{
    ui->setupUi(this);
}

} // View
