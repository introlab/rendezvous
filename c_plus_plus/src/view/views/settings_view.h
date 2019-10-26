#ifndef SETTINGS_VIEW_H
#define SETTINGS_VIEW_H

#include "abstract_view.h"

namespace Ui { class SettingsView; }
namespace Model { class ISettings; }

namespace View
{

class SettingsView : public AbstractView
{
    public:
        explicit SettingsView(Model::ISettings& settings, QWidget *parent = nullptr);

    public slots:
        void onLanguageComboboxCurrentIndexChanged(const int& index);
        void onAutoTranscriptionCheckBoxStateChanged(const int& state);
        void onOutputFolderButtonClicked();

    private:
        Ui::SettingsView *m_ui;
        Model::ISettings &m_settings;
};


} // View

#endif // SETTINGS_VIEW_H
