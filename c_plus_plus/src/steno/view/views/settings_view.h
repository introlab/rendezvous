#ifndef SETTINGS_VIEW_H
#define SETTINGS_VIEW_H

#include "view/views/abstract_view.h"
#include "view/views/struct_appclication_settings.h"

#include<QFileDialog>

namespace Ui { class SettingsView; }

namespace View
{

class SettingsView : public AbstractView
{
    public:
        explicit SettingsView(QWidget *parent = nullptr);
        virtual ~SettingsView();

    public slots:
        void onbtnBrowseDefaultOutputFolderClick();
        void onbtnBrowseCameraConfigFileClick();
        void onbtnBrowseMicrophoneConfigFileClick();
        void onbtnOdasLibraryBrowseClick();
        void ontbtnFaceDetectionMethodBrowseClick();
        void onbtnGoogleServiceAccountBrowseClick();

    private:
        Ui::SettingsView *m_ui;
        QFileDialog m_dialog;
        View::ApplicationSettingsStruct *m_structAppSettings;

};


} // View

#endif // SETTINGS_VIEW_H
