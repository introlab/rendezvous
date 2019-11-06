#ifndef KEYS_H
#define KEYS_H

#include <QCoreApplication>
#include <QDir>
#include <QString>

namespace Model
{
const QString APP_CONFIG_FILE =  QCoreApplication::applicationDirPath() + "/../steno.conf";
const QString MICROPHONE_CONFIGURATION = QCoreApplication::applicationDirPath() + "/../configs/odas/odas_16_mic.cfg";
const QString ODAS_LIBRARY = QCoreApplication::applicationDirPath() + "/../../odas/bin/odaslive";
const QString VIRTUAL_CAMERA_DEVICE = "/dev/video1";

}    // namespace Model

#endif    // KEYS_H
