#ifndef BASE_CONFIG_H
#define BASE_CONFIG_H

#include <memory>

#include <QMap>
#include <QObject>
#include <QSettings>
#include <QString>
#include <QVariant>

#include <QDebug>

namespace Model
{
class BaseConfig : public QObject
{
    Q_OBJECT
   public:
    BaseConfig(std::shared_ptr<QSettings> settings);
    BaseConfig(const QString &group, std::shared_ptr<QSettings> settings);

    inline QString group() const
    {
        return m_group;
    }

    template <typename GroupEnum>
    std::shared_ptr<BaseConfig> subConfig(GroupEnum group);

    template <typename KeyEnum>
    void setValue(KeyEnum key, const QVariant &value);

    template <typename KeyEnum>
    QVariant value(KeyEnum key) const;

   protected:
    void addSubConfig(std::shared_ptr<BaseConfig> cfg);
    void updateSubconfigs();
    virtual void update()
    {
    }

   private:
    QString m_group;
    QMap<QString, std::shared_ptr<BaseConfig>> m_mapConfig;
    QList<std::shared_ptr<BaseConfig>> m_subConfigs;
    std::shared_ptr<QSettings> m_settings;
};

template <typename GroupEnum>
std::shared_ptr<BaseConfig> BaseConfig::subConfig(GroupEnum group)
{
    return m_mapConfig.value(QVariant::fromValue(group).toString());
}

template <typename KeyEnum>
void BaseConfig::setValue(KeyEnum key, const QVariant &value)
{
    m_settings->setValue(m_group + "/" + QVariant::fromValue(key).toString(), value);
}

template <typename KeyEnum>
QVariant BaseConfig::value(KeyEnum key) const
{
    return m_settings->value(m_group + "/" + QVariant::fromValue(key).toString());
}
}    // namespace Model

#endif    // BASE_CONFIG_H
