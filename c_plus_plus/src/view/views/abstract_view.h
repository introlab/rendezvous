#ifndef ABSTRACT_VIEW_H
#define ABSTRACT_VIEW_H

#include <utility>

#include <QWidget>

namespace View
{
class AbstractView : public QWidget
{
    Q_OBJECT

   public:
    explicit AbstractView(QString name, QWidget* parent)
        : QWidget(parent)
        , name(std::move(name))
    {
    }
    const QString& getName()
    {
        return name;
    }

   private:
    QString name;
};

}    // namespace View

#endif    // ABSTRACT_VIEW_H
