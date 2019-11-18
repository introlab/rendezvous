#include "filesutil.h"

#include <QDir>

namespace Model
{
namespace Util
{
/**
 * @brief Verify in a specified directory what is the most recent file edited. Returns fale if there is an error.
 * @param [IN] directory - Folder to check in.
 * @param [IN] extension - extension name to filter.
 * @param [OUT] mostRecentModifiedFilePath - output filepath.
 * @return true/false if success
 */
bool mostRecentModified(const QString& directory, const QString& extension, QString& mostRecentModifiedFilePath)
{
    QDir dir(directory);
    if (!dir.exists()) return false;

    QStringList extensionFilter("*." + extension);
    QFileInfoList list = dir.entryInfoList(extensionFilter, QDir::NoFilter, QDir::Time);
    if (list.isEmpty()) return false;

    mostRecentModifiedFilePath = list.first().absoluteFilePath();
    return true;
}
}    // namespace Util
}    // namespace Model
