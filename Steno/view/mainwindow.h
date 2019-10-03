#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QStackedWidget>

#include "view/components/sidebar.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

    public:
        MainWindow(QWidget *parent = nullptr);
        ~MainWindow();

    private:
        Ui::MainWindow *ui;
        SideBar sideBar;
        QStackedWidget views;

    public slots:
        void onSideBarCurrentRowChanged(int index) {views.setCurrentIndex(index);}

};
#endif // MAINWINDOW_H
