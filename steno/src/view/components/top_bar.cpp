#include "top_bar.h"
#include "ui_top_bar.h"

#include "colors.h"

#include <QStyle>

namespace View
{

TopBar::TopBar(QWidget *parent)
: QWidget(parent)
, m_ui(new Ui::TopBar)
{
    m_ui->setupUi(this);

    QPalette pal = palette();
    pal.setColor(QPalette::Window, LIGHT_GREEN);
    pal.setColor(QPalette::WindowText, WHITE);
    pal.setColor(QPalette::Button, DARK_GREEN);
    pal.setColor(QPalette::ButtonText, WHITE);
    setAutoFillBackground(true);
    setPalette(pal);
}

}
