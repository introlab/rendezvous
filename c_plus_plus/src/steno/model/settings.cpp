#include "settings.h"

#include <QString>

namespace Model
{

Settings::Settings()
    : ISettings()
    , m_settings(m_companyName, m_applicationName)
{

}

} // Model

