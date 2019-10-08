#ifndef SETTINGS_VIEW_H
#define SETTINGS_VIEW_H

#include "model/i_settings_view.h"

namespace Model
{

    class SettingsView : public ISettingsView
    {
        public:
            explicit SettingsView(QWidget *parent = nullptr);
            virtual ~SettingsView() override;
            virtual void openFile() override;






    };


} // Model



#endif // SETTINGS_VIEW_H
