#include "side_bar.h"
#include "ui_side_bar.h"

#include <QListWidgetItem>
#include "side_bar_item.h"

namespace View
{
SideBar::SideBar(QWidget* parent)
    : QWidget(parent)
    , m_ui(new Ui::SideBar)
    , m_itemSize(m_itemWidth, m_itemHeight)
{
    m_ui->setupUi(this);

    setStyleSheet(
        "QListWidget"
        "{"
        "border: none;"
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

    connect(m_ui->list, &QListWidget::currentRowChanged, [=](int index) { emit currentRowChanged(index); });
}
void SideBar::add(const QString& name, const QIcon& icon)
{
    auto sideBarItem = new SideBarItem(name, icon, this);
    auto listWidgetItem = new QListWidgetItem(m_ui->list);
    m_ui->list->addItem(listWidgetItem);
    m_ui->list->setItemWidget(listWidgetItem, sideBarItem);
    listWidgetItem->setSizeHint(m_itemSize);
}

void SideBar::setCurrentRow(int row)
{
    m_ui->list->setCurrentRow(row);
}

}    // namespace View
