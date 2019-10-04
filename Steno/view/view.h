#ifndef VIEW_H
#define VIEW_H

#include <QWidget>

class Views : public QWidget
{
    Q_OBJECT

    public:

        explicit View(const QString &name, QWidget *parent)
            : QWidget(parent)
            , name(name)
        {}

        virtual ~View() = 0;

    private:
        QString name;
};

#endif // VIEW_H
