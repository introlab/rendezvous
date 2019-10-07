#ifndef ABSTRACT_VIEW_H
#define ABSTRACT_VIEW_H

#include <QWidget>

namespace View
{

class AbstractView : public QWidget
{
    Q_OBJECT

    public:

        explicit AbstractView(const QString &name, QWidget *parent)
            : QWidget(parent)
            , name(name)
        {}

        const QString& getName() {return name;}

    private:
        QString name;
};

} // View

#endif // ABSTRACT_VIEW_H
