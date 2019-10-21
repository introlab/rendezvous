#ifndef SETTINGS_H
#define SETTINGS_H

#include "i_settings.h"

class QSettings;
class QString;
class QVariant;

namespace Model
{

class Settings : public ISettings
{
public:
    Settings();
    void set(const QString &key, const QVariant &value) override;
    QVariant get(const QString &key) const override;

private:
    void load();

    QSettings *m_settings;
};

} // Model

#endif // SETTINGS_H
