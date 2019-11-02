#ifndef SRT_FILE_H
#define SRT_FILE_H

#include "subtitle_item.h"

#include <vector>

#include <QString>

namespace Model
{
class SrtFile
{
   public:
    static std::vector<SubtitleItem> parse(const QString &path);
};

}    // namespace Model

#endif    // SRT_FILE_H
