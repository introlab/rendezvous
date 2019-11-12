#ifndef SIDEBAR_H
#define SIDEBAR_H

#include <QListWidget>
#include <QWidget>

namespace Ui
{
class SideBar;
}

namespace View
{
class SideBar : public QWidget
{
    Q_OBJECT

   public:
    SideBar(QWidget* parent = nullptr);

    void add(const QString& name, const QIcon &icon);
    void setCurrentRow(int row);

   signals:
    void currentRowChanged(int index);

   private:
    Ui::SideBar* m_ui;
    const uint8_t m_itemHeight = 100;
    const uint8_t m_itemWidth = 100;
    const QSize m_itemSize;
};

}    // namespace View

#endif    // SIDEBAR_H
