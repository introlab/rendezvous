#ifndef I_RECORDER_H
#define I_RECORDER_H

#include <string>

#include <QWidget>

namespace Model
{

class IRecorder : public QWidget
{
Q_OBJECT

public:
    IRecorder(QWidget *parent = nullptr) : QWidget(parent) {}
    virtual void start(std::string outputPath) = 0;
    virtual void stop() = 0;
};

} // Model

#endif // I_RECORDER_H
