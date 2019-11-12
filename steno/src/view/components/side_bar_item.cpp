#include "side_bar_item.h"
#include "ui_side_bar_item.h"

#include <QTransform>

namespace View
{

SideBarItem::SideBarItem(const QString &name, const QIcon &icon, QWidget *parent)
    : QWidget(parent)
    , m_ui(new Ui::SideBarItem)
{
    m_ui->setupUi(this);

    QPixmap pixmap = icon.pixmap(QSize(48, 48));
    m_ui->icon->setPixmap(pixmap);

    m_ui->name->setText(name);
    m_ui->name->setStyleSheet("color: #ffffff;");
}

}
