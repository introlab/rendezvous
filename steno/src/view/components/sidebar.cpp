#include "sidebar.h"

namespace View
{
SideBar::SideBar(QWidget* parent)
    : QListWidget(parent)
    , m_itemSize(0, m_itemHeight)
{
    setFixedWidth(m_itemWidth);

    setStyleSheet(
    "QListWidget"
    "{"
    "    background-color: #4a4a4a;"
    "    color: #ffffff;"
    "    selection-background-color: #00a559;"
    "    selection-color: #ffffff;"
    "}"
    ""
    "QListWidget::item:hover"
    "{"
    "  color: #ffffff;"
    "  background-color: #00a559;"
    "}");
}
void SideBar::add(const QString& name)
{
    addItem(name);

    for (int i = 0; i < count(); i++)
    {
        item(i)->setSizeHint(m_itemSize);
        item(i)->setTextAlignment(Qt::AlignCenter);
    }
}

}    // namespace View
