/********************************************************************************
** Form generated from reading UI file 'recording.ui'
**
** Created by: Qt User Interface Compiler version 5.12.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_RECORDING_H
#define UI_RECORDING_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Recording
{
public:
    QGridLayout *gridLayout;
    QVBoxLayout *verticalLayout;
    QFrame *virtualCameraFrame;
    QHBoxLayout *horizontalLayout_5;
    QHBoxLayout *horizontalLayout_4;
    QFrame *line_5;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *btnStartStopRecord;

    void setupUi(QWidget *Recording)
    {
        if (Recording->objectName().isEmpty())
            Recording->setObjectName(QString::fromUtf8("Recording"));
        Recording->resize(368, 281);
        Recording->setMouseTracking(false);
        gridLayout = new QGridLayout(Recording);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        virtualCameraFrame = new QFrame(Recording);
        virtualCameraFrame->setObjectName(QString::fromUtf8("virtualCameraFrame"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(virtualCameraFrame->sizePolicy().hasHeightForWidth());
        virtualCameraFrame->setSizePolicy(sizePolicy);
        virtualCameraFrame->setMinimumSize(QSize(0, 0));
        virtualCameraFrame->setFrameShape(QFrame::StyledPanel);
        virtualCameraFrame->setFrameShadow(QFrame::Raised);
        horizontalLayout_5 = new QHBoxLayout(virtualCameraFrame);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));

        horizontalLayout_5->addLayout(horizontalLayout_4);


        verticalLayout->addWidget(virtualCameraFrame);

        line_5 = new QFrame(Recording);
        line_5->setObjectName(QString::fromUtf8("line_5"));
        line_5->setFrameShape(QFrame::HLine);
        line_5->setFrameShadow(QFrame::Sunken);

        verticalLayout->addWidget(line_5);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        btnStartStopRecord = new QPushButton(Recording);
        btnStartStopRecord->setObjectName(QString::fromUtf8("btnStartStopRecord"));
        btnStartStopRecord->setCheckable(false);
        btnStartStopRecord->setChecked(false);

        horizontalLayout->addWidget(btnStartStopRecord);


        verticalLayout->addLayout(horizontalLayout);


        gridLayout->addLayout(verticalLayout, 0, 0, 1, 1);


        retranslateUi(Recording);

        QMetaObject::connectSlotsByName(Recording);
    } // setupUi

    void retranslateUi(QWidget *Recording)
    {
        btnStartStopRecord->setText(QApplication::translate("Recording", "Start Recording", nullptr));
        Q_UNUSED(Recording);
    } // retranslateUi

};

namespace Ui {
    class Recording: public Ui_Recording {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_RECORDING_H
