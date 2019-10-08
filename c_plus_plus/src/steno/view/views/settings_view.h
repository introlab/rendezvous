#ifndef SETTINGS_VIEW_H
#define SETTINGS_VIEW_H

#include "view/views/abstract_view.h"
#include "view/views/struct_appclication_settings.h"

#include<QFileDialog>

namespace Ui { class SettingsView; }

namespace View
{

<<<<<<< HEAD
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
=======
class SettingsView : public AbstractView
{
public:
    explicit SettingsView(QWidget *parent = nullptr);

private:
    Ui::SettingsView *ui;
};
>>>>>>> master

} // View

#endif // SETTINGS_VIEW_H
