#include "base_config.h"

namespace Model
{
BaseConfig::BaseConfig(std::shared_ptr<QSettings> settings)
    : m_settings(settings)
{
}

BaseConfig::BaseConfig(const QString &group, std::shared_ptr<QSettings> settings)
    : m_group(group)
    , m_settings(settings)
{
}

void BaseConfig::updateSubconfigs()
{
    for (auto config : m_subConfigs)
    {
        config->update();
    }
}

void BaseConfig::addSubConfig(std::shared_ptr<BaseConfig> cfg)
{
    if (!cfg->group().isEmpty() && !m_mapConfig.contains(cfg->group()))
    {
        m_mapConfig.insert(cfg->group(), cfg);
        m_subConfigs.push_back(cfg);
    }
}

}    // namespace Model
