#ifndef SIDEBAR_H
#define SIDEBAR_H

#include <QListWidget>

class SideBar : public QListWidget
{
    Q_OBJECT

    public:
        SideBar(QWidget *parent = nullptr);

        void add(QString name);

    private:
        const QSize itemSize;
};

#endif // SIDEBAR_H
