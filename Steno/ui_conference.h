/********************************************************************************
** Form generated from reading UI file 'conference.ui'
**
** Created by: Qt User Interface Compiler version 5.12.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CONFERENCE_H
#define UI_CONFERENCE_H

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

class Ui_Conference
{
public:
    QGridLayout *gridLayout;
    QVBoxLayout *verticalLayout;
    QFrame *virtualCameraFrame;
    QHBoxLayout *horizontalLayout_6;
    QSpacerItem *horizontalSpacer_3;
    QSpacerItem *horizontalSpacer_4;
    QPushButton *btnStartStopVideo;
    QFrame *soundPositionsFrame;
    QHBoxLayout *horizontalLayout_2;
    QVBoxLayout *soundPositionsLayout;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *btnStartStopOdas;

    void setupUi(QWidget *Conference)
    {
        if (Conference->objectName().isEmpty())
            Conference->setObjectName(QString::fromUtf8("Conference"));
        Conference->resize(769, 718);
        Conference->setMouseTracking(false);
        gridLayout = new QGridLayout(Conference);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        virtualCameraFrame = new QFrame(Conference);
        virtualCameraFrame->setObjectName(QString::fromUtf8("virtualCameraFrame"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(virtualCameraFrame->sizePolicy().hasHeightForWidth());
        virtualCameraFrame->setSizePolicy(sizePolicy);
        virtualCameraFrame->setMinimumSize(QSize(0, 300));
        virtualCameraFrame->setFrameShape(QFrame::StyledPanel);
        virtualCameraFrame->setFrameShadow(QFrame::Raised);

        verticalLayout->addWidget(virtualCameraFrame);

        horizontalLayout_6 = new QHBoxLayout();
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        horizontalLayout_6->setContentsMargins(6, 6, 6, 6);
        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_3);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_4);

        btnStartStopVideo = new QPushButton(Conference);
        btnStartStopVideo->setObjectName(QString::fromUtf8("btnStartStopVideo"));

        horizontalLayout_6->addWidget(btnStartStopVideo);


        verticalLayout->addLayout(horizontalLayout_6);

        soundPositionsFrame = new QFrame(Conference);
        soundPositionsFrame->setObjectName(QString::fromUtf8("soundPositionsFrame"));
        soundPositionsFrame->setMinimumSize(QSize(0, 300));
        soundPositionsFrame->setMaximumSize(QSize(16777215, 400));
        soundPositionsFrame->setFrameShape(QFrame::NoFrame);
        soundPositionsFrame->setFrameShadow(QFrame::Raised);
        horizontalLayout_2 = new QHBoxLayout(soundPositionsFrame);
        horizontalLayout_2->setSpacing(0);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(0, 0, 0, 0);
        soundPositionsLayout = new QVBoxLayout();
        soundPositionsLayout->setSpacing(6);
        soundPositionsLayout->setObjectName(QString::fromUtf8("soundPositionsLayout"));
        soundPositionsLayout->setContentsMargins(6, 6, 6, 6);

        horizontalLayout_2->addLayout(soundPositionsLayout);


        verticalLayout->addWidget(soundPositionsFrame);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(6, 6, 6, 6);
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        btnStartStopOdas = new QPushButton(Conference);
        btnStartStopOdas->setObjectName(QString::fromUtf8("btnStartStopOdas"));

        horizontalLayout->addWidget(btnStartStopOdas);


        verticalLayout->addLayout(horizontalLayout);


        gridLayout->addLayout(verticalLayout, 0, 0, 1, 1);


        retranslateUi(Conference);

        QMetaObject::connectSlotsByName(Conference);
    } // setupUi

    void retranslateUi(QWidget *Conference)
    {
        btnStartStopVideo->setText(QApplication::translate("Conference", "Start Video", nullptr));
#ifndef QT_NO_TOOLTIP
        btnStartStopOdas->setToolTip(QString());
#endif // QT_NO_TOOLTIP
        btnStartStopOdas->setText(QApplication::translate("Conference", "Start ODAS", nullptr));
        Q_UNUSED(Conference);
    } // retranslateUi

};

namespace Ui {
    class Conference: public Ui_Conference {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONFERENCE_H
