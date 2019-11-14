#include "settings_view.h"
#include "ui_settings_view.h"

#include "model/app_config.h"
#include "model/transcription/transcription.h"
#include "model/transcription/transcription_config.h"

#include <QComboBox>
#include <QFileDialog>

namespace View
{
SettingsView::SettingsView(std::shared_ptr<Model::Config> config, QWidget* parent)
    : AbstractView("Settings", parent)
    , m_ui(new Ui::SettingsView)
    , m_config(config)
{
    m_ui->setupUi(this);

    for (auto i = 0; i != Model::Transcription::Language::COUNT; i++)
    {
        m_ui->languageComboBox->addItem(
            Model::Transcription::languageName(static_cast<Model::Transcription::Language>(i)));
    }

    m_ui->outputFolderLineEdit->setText(
        m_config->subConfig(Model::Config::APP)->value(Model::AppConfig::OUTPUT_FOLDER).toString());
    m_ui->languageComboBox->setCurrentIndex(
        m_config->subConfig(Model::Config::TRANSCRIPTION)->value(Model::TranscriptionConfig::LANGUAGE).toInt());
    m_ui->autoTranscriptionCheckBox->setChecked(m_config->subConfig(Model::Config::TRANSCRIPTION)
                                                    ->value(Model::TranscriptionConfig::AUTOMATIC_TRANSCRIPTION)
                                                    .toBool());

    connect(m_ui->outputFolderButton, &QAbstractButton::clicked, [=] { onOutputFolderButtonClicked(); });
    connect(m_ui->languageComboBox, qOverload<int>(&QComboBox::currentIndexChanged),
            [=](const int& index) { onLanguageComboboxCurrentIndexChanged(index); });
    connect(m_ui->autoTranscriptionCheckBox, &QCheckBox::stateChanged,
            [=](const int& state) { onAutoTranscriptionCheckBoxStateChanged(state); });
}

void SettingsView::onOutputFolderButtonClicked()
{
    QString outputFolder = QFileDialog::getExistingDirectory(
        this, "Output Folder",
        m_config->subConfig(Model::Config::APP)->value(Model::AppConfig::OUTPUT_FOLDER).toString(),
        QFileDialog::ShowDirsOnly);
    if (!outputFolder.isEmpty())
    {
        m_config->subConfig(Model::Config::APP)->setValue(Model::AppConfig::OUTPUT_FOLDER, outputFolder);
        m_ui->outputFolderLineEdit->setText(outputFolder);
    }
}

void SettingsView::onLanguageComboboxCurrentIndexChanged(const int& index)
{
    m_config->subConfig(Model::Config::TRANSCRIPTION)
        ->setValue(Model::TranscriptionConfig::LANGUAGE, static_cast<Model::Transcription::Language>(index));
}

void SettingsView::onAutoTranscriptionCheckBoxStateChanged(const int& state)
{
    m_config->subConfig(Model::Config::TRANSCRIPTION)
        ->setValue(Model::TranscriptionConfig::AUTOMATIC_TRANSCRIPTION, state == Qt::Checked);
}

}    // namespace View
