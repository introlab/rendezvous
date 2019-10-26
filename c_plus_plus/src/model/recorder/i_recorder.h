#ifndef I_RECORDER_H
#define I_RECORDER_H

#include <QWidget>

namespace Model
{
class IRecorder : public QWidget
{
    Q_OBJECT

   public:
    IRecorder(QWidget *parent = nullptr) : QWidget(parent) {}
    virtual void start(const QString outputPath) = 0;
    virtual void stop() = 0;
};

}    // namespace Model

#endif    // I_RECORDER_H
