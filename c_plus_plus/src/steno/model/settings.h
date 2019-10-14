#ifndef SETTINGS_H
#define SETTINGS_H

#include "i_settings.h"

#include <QSettings>

namespace Model
{

class Settings : ISettings
{
public:
    Settings();

private:
    const QString m_companyName = "RendezVous";
    const QString m_applicationName = "Steno";
    QSettings m_settings;
};

} // Model

#endif // SETTINGS_H
