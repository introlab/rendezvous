#include "settings.h"

#include "settings_constants.h"

#include <QDir>
#include <QSettings>

namespace Model
{
Settings::Settings() : m_settings(new QSettings("RendezVous", "Steno"))
{
    load();
}

void Settings::set(const QString &key, const QVariant &value)
{
    m_settings->setValue(key, value);
}

QVariant Settings::get(const QString &key) const
{
    return m_settings->value(key);
}

void Settings::load()
{
    if (!m_settings->value(General::keyName(General::Key::OUTPUT_FOLDER))
             .isValid())
    {
        set(General::keyName(General::Key::OUTPUT_FOLDER), QDir::homePath());
    }

    if (!m_settings->value(Transcription::keyName(Transcription::Key::LANGUAGE))
             .isValid())
    {
        set(Transcription::keyName(Transcription::Key::LANGUAGE),
            Transcription::Language::FR_CA);
    }

    if (!m_settings
             ->value(Transcription::keyName(
                 Transcription::Key::AUTOMATIC_TRANSCRIPTION))
             .isValid())
    {
        set(Transcription::keyName(Transcription::Key::AUTOMATIC_TRANSCRIPTION),
            false);
    }
}

}    // Model
