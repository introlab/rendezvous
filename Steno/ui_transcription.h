/********************************************************************************
** Form generated from reading UI file 'transcription.ui'
**
** Created by: Qt User Interface Compiler version 5.12.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TRANSCRIPTION_H
#define UI_TRANSCRIPTION_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Transcription
{
public:
    QGridLayout *gridLayout;
    QPushButton *btnTranscribe;
    QGroupBox *groupBox_3;
    QVBoxLayout *verticalLayout;
    QTextBrowser *transcriptionResult;
    QSpacerItem *horizontalSpacer;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_2;
    QLineEdit *audioDataPath;
    QPushButton *btnImportAudio;
    QLabel *label;

    void setupUi(QWidget *Transcription)
    {
        if (Transcription->objectName().isEmpty())
            Transcription->setObjectName(QString::fromUtf8("Transcription"));
        Transcription->resize(626, 541);
        gridLayout = new QGridLayout(Transcription);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        btnTranscribe = new QPushButton(Transcription);
        btnTranscribe->setObjectName(QString::fromUtf8("btnTranscribe"));

        gridLayout->addWidget(btnTranscribe, 3, 1, 1, 1);

        groupBox_3 = new QGroupBox(Transcription);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        verticalLayout = new QVBoxLayout(groupBox_3);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        transcriptionResult = new QTextBrowser(groupBox_3);
        transcriptionResult->setObjectName(QString::fromUtf8("transcriptionResult"));
        transcriptionResult->setContextMenuPolicy(Qt::PreventContextMenu);

        verticalLayout->addWidget(transcriptionResult);


        gridLayout->addWidget(groupBox_3, 1, 0, 1, 2);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 3, 0, 1, 1);

        groupBox = new QGroupBox(Transcription);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        gridLayout_2 = new QGridLayout(groupBox);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        audioDataPath = new QLineEdit(groupBox);
        audioDataPath->setObjectName(QString::fromUtf8("audioDataPath"));

        gridLayout_2->addWidget(audioDataPath, 0, 2, 1, 1);

        btnImportAudio = new QPushButton(groupBox);
        btnImportAudio->setObjectName(QString::fromUtf8("btnImportAudio"));

        gridLayout_2->addWidget(btnImportAudio, 0, 1, 1, 1);

        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy);

        gridLayout_2->addWidget(label, 0, 0, 1, 1);


        gridLayout->addWidget(groupBox, 0, 0, 1, 2);


        retranslateUi(Transcription);

        QMetaObject::connectSlotsByName(Transcription);
    } // setupUi

    void retranslateUi(QWidget *Transcription)
    {
        btnTranscribe->setText(QApplication::translate("Transcription", "Transcribe", nullptr));
        groupBox_3->setTitle(QApplication::translate("Transcription", "Result", nullptr));
        transcriptionResult->setHtml(QApplication::translate("Transcription", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", nullptr));
        groupBox->setTitle(QApplication::translate("Transcription", "Input", nullptr));
        btnImportAudio->setText(QApplication::translate("Transcription", "Import", nullptr));
        label->setText(QApplication::translate("Transcription", "Audio Data", nullptr));
        Q_UNUSED(Transcription);
    } // retranslateUi

};

namespace Ui {
    class Transcription: public Ui_Transcription {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TRANSCRIPTION_H
