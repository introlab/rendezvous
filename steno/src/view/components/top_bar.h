#ifndef TOPBAR_H
#define TOPBAR_H

#include "model/stream/i_stream.h"
#include "model/recorder/i_recorder.h"

#include <memory>

#include <QWidget>

namespace Ui
{
class TopBar;
}

namespace View
{
class TopBar : public QWidget
{
    Q_OBJECT

   public:
    TopBar(std::shared_ptr<Model::IStream> stream, std::shared_ptr<Model::IRecorder> recorder, QWidget* parent = nullptr);

   private slots:
    void onStreamStateChanged(const Model::IStream::State& state);
    void onStartButtonClicked();
    void onRecorderStateChanged(const Model::IRecorder::State& state);
    void onRecordButtonClicked();

   private:
    Ui::TopBar* m_ui;
    std::shared_ptr<Model::IStream> m_stream;
    std::shared_ptr<Model::IRecorder> m_recorder;
};

}    // namespace View

#endif    // TOPBAR_H
