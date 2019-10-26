#include "sidebar.h"

namespace View
{
SideBar::SideBar(QWidget* parent) : QListWidget(parent), m_itemSize(0, m_itemHeight) { setFixedWidth(m_itemWidth); }
void SideBar::add(const QString& name)
{
    addItem(name);

    for (int i = 0; i < count(); i++)
    {
        item(i)->setSizeHint(m_itemSize);
    }
}

}    // namespace View
