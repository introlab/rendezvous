#ifndef TOPBAR_H
#define TOPBAR_H

#include <QWidget>

namespace Ui
{
class TopBar;
}

namespace View
{
class TopBar : public QWidget
{
    Q_OBJECT

   public:
    TopBar(QWidget* parent = nullptr);

   private:
    Ui::TopBar* m_ui;
};

}    // namespace View

#endif    // TOPBAR_H
