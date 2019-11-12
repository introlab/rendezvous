#ifndef SIDEBARITEM_H
#define SIDEBARITEM_H

#include <QIcon>
#include <QWidget>

namespace Ui
{
class SideBarItem;
}

namespace View
{
class SideBarItem : public QWidget
{
    Q_OBJECT

   public:
    SideBarItem(const QString& name, const QIcon& icon, QWidget* parent = nullptr);

   private:
    Ui::SideBarItem* m_ui;
};

}    // namespace View

#endif    // SIDEBARITEM_H
