#ifndef SETTINGS_VIEW_H
#define SETTINGS_VIEW_H

#include "abstract_view.h"
#include "model/config/config.h"

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
    explicit SettingsView(std::shared_ptr<Model::Config> config, QWidget* parent = nullptr);

   public slots:
    void onLanguageComboboxCurrentIndexChanged(const int& index);
    void onAutoTranscriptionCheckBoxStateChanged(const int& state);
    void onOutputFolderButtonClicked();

   private:
    Ui::SettingsView* m_ui;
    std::shared_ptr<Model::Config> m_config;
};

}    // namespace View

#endif    // SETTINGS_VIEW_H
