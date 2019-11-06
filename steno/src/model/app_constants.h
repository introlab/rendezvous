#ifndef KEYS_H
#define KEYS_H

#include <QDir>
#include <QString>

namespace Model
{
const QString CAMERA_CONFIGURATION = QDir::homePath();
const QString MICROPHONE_CONFIGURATION =
    QDir::homePath() + "/dev/workspace/rendezvous/steno/configs/odas/odas_16_mic.cfg";
const QString ODAS_LIBRARY = QDir::homePath() + "/dev/lib/odas/bin/odaslive";
const QString GOOGLE_SERVICE_ACCOUNT_FILE = QDir::homePath();
const QString VIRTUAL_CAMERA_DEVICE = "/dev/video1";
const QString APP_CONFIG_FILE = QDir::homePath() + "/dev/workspace/rendezvous/steno/steno.conf";

}    // namespace Model

#endif    // KEYS_H
