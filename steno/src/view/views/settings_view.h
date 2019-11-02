#ifndef SETTINGS_VIEW_H
#define SETTINGS_VIEW_H

#include "abstract_view.h"
#include "model/settings/i_settings.h"

#include <memory>

namespace Ui
{
class SettingsView;
}

namespace View
{
class SettingsView : public AbstractView
{
   public:
    explicit SettingsView(std::shared_ptr<Model::ISettings> settings, QWidget* parent = nullptr);

   public slots:
    void onLanguageComboboxCurrentIndexChanged(const int& index);
    void onAutoTranscriptionCheckBoxStateChanged(const int& state);
    void onOutputFolderButtonClicked();

   private:
    Ui::SettingsView* m_ui;
    std::shared_ptr<Model::ISettings> m_settings;
};

}    // namespace View

#endif    // SETTINGS_VIEW_H
