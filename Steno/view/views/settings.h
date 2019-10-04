#ifndef SETTINGS_H
#define SETTINGS_H

#include <QWidget>

namespace Ui { class Settings; }

namespace View
{

class Settings : public QWidget
{
    Q_OBJECT

    public:
        explicit Settings(QWidget *parent = nullptr);
        virtual ~Settings();

    private:
        Ui::Settings *ui;
};

} // View

#endif // SETTINGS_H
