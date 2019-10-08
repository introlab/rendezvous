/********************************************************************************
** Form generated from reading UI file 'playback.ui'
**
** Created by: Qt User Interface Compiler version 5.12.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PLAYBACK_H
#define UI_PLAYBACK_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Playback
{
public:
    QGridLayout *gridLayout;
    QVBoxLayout *verticalLayout_3;
    QLabel *mediaPlaying;
    QFrame *videoFrame;
    QSlider *mediaPositionSlider;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *importMediaBtn;
    QCheckBox *subtitleCheckBox;
    QSpacerItem *horizontalSpacer;
    QPushButton *playPauseBtn;
    QPushButton *stopBtn;
    QSlider *volumeSlider;

    void setupUi(QWidget *Playback)
    {
        if (Playback->objectName().isEmpty())
            Playback->setObjectName(QString::fromUtf8("Playback"));
        Playback->resize(810, 538);
        Playback->setAutoFillBackground(true);
        gridLayout = new QGridLayout(Playback);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        mediaPlaying = new QLabel(Playback);
        mediaPlaying->setObjectName(QString::fromUtf8("mediaPlaying"));
        mediaPlaying->setAlignment(Qt::AlignCenter);

        verticalLayout_3->addWidget(mediaPlaying);

        videoFrame = new QFrame(Playback);
        videoFrame->setObjectName(QString::fromUtf8("videoFrame"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(videoFrame->sizePolicy().hasHeightForWidth());
        videoFrame->setSizePolicy(sizePolicy);
        videoFrame->setAutoFillBackground(true);
        videoFrame->setFrameShape(QFrame::StyledPanel);
        videoFrame->setFrameShadow(QFrame::Raised);

        verticalLayout_3->addWidget(videoFrame);

        mediaPositionSlider = new QSlider(Playback);
        mediaPositionSlider->setObjectName(QString::fromUtf8("mediaPositionSlider"));
        mediaPositionSlider->setMaximum(1000);
        mediaPositionSlider->setOrientation(Qt::Horizontal);

        verticalLayout_3->addWidget(mediaPositionSlider);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        importMediaBtn = new QPushButton(Playback);
        importMediaBtn->setObjectName(QString::fromUtf8("importMediaBtn"));

        horizontalLayout_3->addWidget(importMediaBtn);

        subtitleCheckBox = new QCheckBox(Playback);
        subtitleCheckBox->setObjectName(QString::fromUtf8("subtitleCheckBox"));
        subtitleCheckBox->setEnabled(false);
        subtitleCheckBox->setChecked(false);

        horizontalLayout_3->addWidget(subtitleCheckBox);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer);

        playPauseBtn = new QPushButton(Playback);
        playPauseBtn->setObjectName(QString::fromUtf8("playPauseBtn"));

        horizontalLayout_3->addWidget(playPauseBtn);

        stopBtn = new QPushButton(Playback);
        stopBtn->setObjectName(QString::fromUtf8("stopBtn"));

        horizontalLayout_3->addWidget(stopBtn);

        volumeSlider = new QSlider(Playback);
        volumeSlider->setObjectName(QString::fromUtf8("volumeSlider"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(volumeSlider->sizePolicy().hasHeightForWidth());
        volumeSlider->setSizePolicy(sizePolicy1);
        volumeSlider->setMaximum(100);
        volumeSlider->setOrientation(Qt::Horizontal);

        horizontalLayout_3->addWidget(volumeSlider);


        verticalLayout_3->addLayout(horizontalLayout_3);


        gridLayout->addLayout(verticalLayout_3, 0, 1, 1, 1);


        retranslateUi(Playback);

        QMetaObject::connectSlotsByName(Playback);
    } // setupUi

    void retranslateUi(QWidget *Playback)
    {
        mediaPlaying->setText(QApplication::translate("Playback", "No media playing", nullptr));
        importMediaBtn->setText(QApplication::translate("Playback", "Video", nullptr));
        subtitleCheckBox->setText(QApplication::translate("Playback", "Subtitle", nullptr));
        playPauseBtn->setText(QApplication::translate("Playback", "Play", nullptr));
        stopBtn->setText(QApplication::translate("Playback", "Stop", nullptr));
        Q_UNUSED(Playback);
    } // retranslateUi

};

namespace Ui {
    class Playback: public Ui_Playback {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PLAYBACK_H
