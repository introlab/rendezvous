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

        void add(QString name);

    private:
        const QSize itemSize;
};

} // View

#endif // SIDEBAR_H
