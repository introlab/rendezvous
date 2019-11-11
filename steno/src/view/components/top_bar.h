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

}

#endif // TOPBAR_H
