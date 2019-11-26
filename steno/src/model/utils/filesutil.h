#ifndef FILESUTIL_H
#define FILESUTIL_H

#include <QString>

namespace Model
{
namespace Util
{
bool mostRecentModified(const QString& directory, const QString& extension, QString& mostRecentModifiedFilePath);
}
}    // namespace Model

#endif    // FILESUTIL_H
