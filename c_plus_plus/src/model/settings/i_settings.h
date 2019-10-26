#ifndef I_SETTINGS_H
#define I_SETTINGS_H

class QString;
class QVariant;

namespace Model
{
class ISettings
{
   public:
    virtual ~ISettings() = default;
    virtual void set(const QString &key, const QVariant &value) = 0;
    virtual QVariant get(const QString &key) const = 0;
};
}

#endif    // I_SETTINGS_H
