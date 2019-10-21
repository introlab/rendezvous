#include "settings_view.h"
#include "ui_settings_view.h"

#include "model/settings/i_settings.h"
#include "model/settings/settings_constants.h"

#include <QComboBox>
#include <QtGlobal>

namespace View
{

SettingsView::SettingsView(Model::ISettings& settings, QWidget *parent)
        : AbstractView("Settings", parent)
        , m_ui(new Ui::SettingsView)
        , m_dialog(this)
        , m_settings(settings)
{
    m_ui->setupUi(this);

    for(auto i = 0; i != Model::Transcription::Language::COUNT; i++)
    {
        m_ui->languageComboBox->addItem(Model::Transcription::languageName(static_cast<Model::Transcription::Language>(i)));
    }

    m_ui->outputFolderLineEdit->setText(m_settings.get(Model::General::keyName(Model::General::Key::OUTPUT_FOLDER)).toString());
    m_ui->languageComboBox->setCurrentIndex(m_settings.get(Model::Transcription::keyName(Model::Transcription::Key::LANGUAGE)).toInt());
    m_ui->autoTranscriptionCheckBox->setChecked(m_settings.get(Model::Transcription::keyName(Model::Transcription::Key::AUTOMATIC_TRANSCRIPTION)).toBool());

    connect(m_ui->outputFolderButton, &QAbstractButton::clicked, [=]{ onOutputFolderButtonClicked(); });
    connect(m_ui->languageComboBox, qOverload<int>(&QComboBox::currentIndexChanged), [=]( const int& index ) { onLanguageComboboxCurrentIndexChanged(index); });
    connect(m_ui->autoTranscriptionCheckBox, &QCheckBox::stateChanged, [=]( const int& state ) { onAutoTranscriptionCheckBoxStateChanged(state); });
}

void SettingsView::onOutputFolderButtonClicked()
{
    QString outputFolder = m_dialog.getExistingDirectory(this,
                                                         "Output Folder",
                                                         m_settings.get(Model::General::keyName(Model::General::Key::OUTPUT_FOLDER)).toString(),
                                                         QFileDialog::ShowDirsOnly);
    if (!outputFolder.isEmpty())
    {
        m_settings.set(Model::General::keyName(Model::General::Key::OUTPUT_FOLDER), outputFolder);
    }
}

void SettingsView::onLanguageComboboxCurrentIndexChanged(const int& index)
{
    m_settings.set(Model::Transcription::keyName(Model::Transcription::Key::LANGUAGE),
                   static_cast<Model::Transcription::Language>(index));
}

void SettingsView::onAutoTranscriptionCheckBoxStateChanged(const int& state)
{
    m_settings.set(Model::Transcription::keyName(Model::Transcription::Key::AUTOMATIC_TRANSCRIPTION),
                   state == Qt::Checked ? true : false);
}

} // View
