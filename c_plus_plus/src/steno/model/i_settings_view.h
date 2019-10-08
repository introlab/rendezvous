#ifndef I_SETTINGS_VIEW_H
#define I_SETTINGS_VIEW_H

#include <QWidget>

namespace Model
{

    class ISettingsView : public QWidget
    {
        Q_OBJECT

        public:
            ISettingsView(QWidget *parent = nullptr) : QWidget(parent){}
            virtual ~ISettingsView(){}
            virtual void openFile() = 0;


    };


}

#endif // I_SETTINGS_VIEW_H
