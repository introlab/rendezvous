/********************************************************************************
** Form generated from reading UI file 'audio_processing.ui'
**
** Created by: Qt User Interface Compiler version 5.12.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_AUDIO_PROCESSING_H
#define UI_AUDIO_PROCESSING_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_AudioProcessing
{
public:
    QGridLayout *gridLayout;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_2;
    QLabel *label_2;
    QPushButton *btnProcessAudio;
    QPushButton *btnImportAudio;
    QSpacerItem *horizontalSpacer;
    QLabel *label;
    QComboBox *cbNoiseReductionLib;
    QLineEdit *audioDataPath;
    QLabel *lblState;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *AudioProcessing)
    {
        if (AudioProcessing->objectName().isEmpty())
            AudioProcessing->setObjectName(QString::fromUtf8("AudioProcessing"));
        AudioProcessing->resize(626, 541);
        gridLayout = new QGridLayout(AudioProcessing);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        groupBox = new QGroupBox(AudioProcessing);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        gridLayout_2 = new QGridLayout(groupBox);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label_2->sizePolicy().hasHeightForWidth());
        label_2->setSizePolicy(sizePolicy);

        gridLayout_2->addWidget(label_2, 1, 0, 1, 1);

        btnProcessAudio = new QPushButton(groupBox);
        btnProcessAudio->setObjectName(QString::fromUtf8("btnProcessAudio"));

        gridLayout_2->addWidget(btnProcessAudio, 4, 3, 1, 1);

        btnImportAudio = new QPushButton(groupBox);
        btnImportAudio->setObjectName(QString::fromUtf8("btnImportAudio"));

        gridLayout_2->addWidget(btnImportAudio, 2, 1, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer, 4, 2, 1, 1);

        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));
        sizePolicy.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy);

        gridLayout_2->addWidget(label, 2, 0, 1, 1);

        cbNoiseReductionLib = new QComboBox(groupBox);
        cbNoiseReductionLib->setObjectName(QString::fromUtf8("cbNoiseReductionLib"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(cbNoiseReductionLib->sizePolicy().hasHeightForWidth());
        cbNoiseReductionLib->setSizePolicy(sizePolicy1);

        gridLayout_2->addWidget(cbNoiseReductionLib, 1, 1, 1, 1);

        audioDataPath = new QLineEdit(groupBox);
        audioDataPath->setObjectName(QString::fromUtf8("audioDataPath"));

        gridLayout_2->addWidget(audioDataPath, 2, 2, 1, 2);

        lblState = new QLabel(groupBox);
        lblState->setObjectName(QString::fromUtf8("lblState"));
        sizePolicy.setHeightForWidth(lblState->sizePolicy().hasHeightForWidth());
        lblState->setSizePolicy(sizePolicy);

        gridLayout_2->addWidget(lblState, 5, 3, 1, 1);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer, 6, 3, 1, 1);


        gridLayout->addWidget(groupBox, 0, 0, 1, 1);


        retranslateUi(AudioProcessing);

        QMetaObject::connectSlotsByName(AudioProcessing);
    } // setupUi

    void retranslateUi(QWidget *AudioProcessing)
    {
        AudioProcessing->setWindowTitle(QApplication::translate("AudioProcessing", "Form", nullptr));
        groupBox->setTitle(QApplication::translate("AudioProcessing", "Noise Reduction", nullptr));
        label_2->setText(QApplication::translate("AudioProcessing", "Library", nullptr));
        btnProcessAudio->setText(QApplication::translate("AudioProcessing", "Process Audio", nullptr));
        btnImportAudio->setText(QApplication::translate("AudioProcessing", "Import", nullptr));
        label->setText(QApplication::translate("AudioProcessing", "Audio Data", nullptr));
        lblState->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class AudioProcessing: public Ui_AudioProcessing {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_AUDIO_PROCESSING_H
