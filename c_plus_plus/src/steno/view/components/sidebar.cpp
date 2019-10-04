#include "sidebar.h"

namespace View
{

SideBar::SideBar(QWidget *parent)
    : QListWidget(parent)
    , itemSize(QSize(0, 40))
{
    setFixedWidth(150);
}

void SideBar::add(QString name)
{
    addItem(name);

    for (int i = 0; i < count(); i++)
    {
        item(i)->setSizeHint(itemSize);
    }
}

} // View
