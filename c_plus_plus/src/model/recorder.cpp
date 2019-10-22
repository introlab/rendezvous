#include "model/recorder.h"

#include <QMediaRecorder>
#include <QCamera>
#include <QCameraInfo>

namespace Model
{

Recorder::Recorder(QWidget *parent)
    : IRecorder(parent)
    , m_camera(new QCamera(getCameraInfo()))
{
    m_mediaRecorder = new QMediaRecorder(m_camera);
}

QCameraInfo Recorder::getCameraInfo()
{

}

}
