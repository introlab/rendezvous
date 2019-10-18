#ifndef SIDEBAR_H
#define SIDEBAR_H

#include <QListWidget>

namespace View
{

class SideBar : public QListWidget
{
Q_OBJECT

public:
    SideBar(QWidget *parent = nullptr);

    void add(const QString& name);

private:
    const uint8_t m_itemHeight = 40;
    const uint8_t m_itemWidth = 150;
    const QSize m_itemSize;
};

} // View

#endif // SIDEBAR_H
