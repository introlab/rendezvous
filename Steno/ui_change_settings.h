/********************************************************************************
** Form generated from reading UI file 'change_settings.ui'
**
** Created by: Qt User Interface Compiler version 5.12.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CHANGE_SETTINGS_H
#define UI_CHANGE_SETTINGS_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ChangeSettings
{
public:
    QGridLayout *gridLayout;
    QGroupBox *groupBox_5;
    QGridLayout *gridLayout_7;
    QLabel *label_4;
    QLineEdit *serviceAccountPath;
    QCheckBox *enhancedCheckBox;
    QLabel *label_7;
    QLabel *label_12;
    QLabel *label_10;
    QComboBox *languageComboBox;
    QCheckBox *autoTranscriptionCheckBox;
    QComboBox *modelComboBox;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *btnBrowseServiceAccount;
    QLabel *label_8;
    QComboBox *encodingComboBox;
    QLabel *label_11;
    QSpinBox *sampleRateSpinBox;
    QLabel *label_9;
    QLabel *label_13;
    QSpinBox *channelCountSpinBox;
    QSpacerItem *verticalSpacer;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_6;
    QPushButton *btnBrowseOdas;
    QLabel *label_6;
    QLabel *label_2;
    QLineEdit *odasPath;
    QLineEdit *cameraConfigPath;
    QLabel *label;
    QLineEdit *micConfigPath;
    QLabel *label_3;
    QPushButton *btnBrowseMicConfig;
    QPushButton *btnBrowseCameraConfig;
    QSpacerItem *horizontalSpacer;
    QComboBox *cbFaceDetection;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_5;
    QLineEdit *defaultOutputFolder;
    QPushButton *btnBrowseDefaultOutputFolder;
    QLabel *label_5;
    QSpacerItem *horizontalSpacer_3;

    void setupUi(QWidget *ChangeSettings)
    {
        if (ChangeSettings->objectName().isEmpty())
            ChangeSettings->setObjectName(QString::fromUtf8("ChangeSettings"));
        ChangeSettings->resize(1145, 617);
        gridLayout = new QGridLayout(ChangeSettings);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        groupBox_5 = new QGroupBox(ChangeSettings);
        groupBox_5->setObjectName(QString::fromUtf8("groupBox_5"));
        gridLayout_7 = new QGridLayout(groupBox_5);
        gridLayout_7->setSpacing(6);
        gridLayout_7->setContentsMargins(11, 11, 11, 11);
        gridLayout_7->setObjectName(QString::fromUtf8("gridLayout_7"));
        gridLayout_7->setHorizontalSpacing(6);
        label_4 = new QLabel(groupBox_5);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setMinimumSize(QSize(203, 0));

        gridLayout_7->addWidget(label_4, 0, 0, 1, 2);

        serviceAccountPath = new QLineEdit(groupBox_5);
        serviceAccountPath->setObjectName(QString::fromUtf8("serviceAccountPath"));
        serviceAccountPath->setEnabled(false);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(serviceAccountPath->sizePolicy().hasHeightForWidth());
        serviceAccountPath->setSizePolicy(sizePolicy);
        serviceAccountPath->setMinimumSize(QSize(540, 0));

        gridLayout_7->addWidget(serviceAccountPath, 0, 2, 1, 1);

        enhancedCheckBox = new QCheckBox(groupBox_5);
        enhancedCheckBox->setObjectName(QString::fromUtf8("enhancedCheckBox"));

        gridLayout_7->addWidget(enhancedCheckBox, 8, 2, 1, 1);

        label_7 = new QLabel(groupBox_5);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout_7->addWidget(label_7, 7, 0, 1, 1);

        label_12 = new QLabel(groupBox_5);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        gridLayout_7->addWidget(label_12, 8, 0, 1, 1);

        label_10 = new QLabel(groupBox_5);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        gridLayout_7->addWidget(label_10, 4, 0, 1, 1);

        languageComboBox = new QComboBox(groupBox_5);
        languageComboBox->setObjectName(QString::fromUtf8("languageComboBox"));

        gridLayout_7->addWidget(languageComboBox, 3, 2, 1, 1);

        autoTranscriptionCheckBox = new QCheckBox(groupBox_5);
        autoTranscriptionCheckBox->setObjectName(QString::fromUtf8("autoTranscriptionCheckBox"));

        gridLayout_7->addWidget(autoTranscriptionCheckBox, 7, 2, 1, 1);

        modelComboBox = new QComboBox(groupBox_5);
        modelComboBox->setObjectName(QString::fromUtf8("modelComboBox"));

        gridLayout_7->addWidget(modelComboBox, 4, 2, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_7->addItem(horizontalSpacer_2, 0, 4, 1, 1);

        btnBrowseServiceAccount = new QPushButton(groupBox_5);
        btnBrowseServiceAccount->setObjectName(QString::fromUtf8("btnBrowseServiceAccount"));

        gridLayout_7->addWidget(btnBrowseServiceAccount, 0, 3, 1, 1);

        label_8 = new QLabel(groupBox_5);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout_7->addWidget(label_8, 2, 0, 1, 1);

        encodingComboBox = new QComboBox(groupBox_5);
        encodingComboBox->setObjectName(QString::fromUtf8("encodingComboBox"));

        gridLayout_7->addWidget(encodingComboBox, 2, 2, 1, 1);

        label_11 = new QLabel(groupBox_5);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        gridLayout_7->addWidget(label_11, 6, 0, 1, 1);

        sampleRateSpinBox = new QSpinBox(groupBox_5);
        sampleRateSpinBox->setObjectName(QString::fromUtf8("sampleRateSpinBox"));

        gridLayout_7->addWidget(sampleRateSpinBox, 6, 2, 1, 1);

        label_9 = new QLabel(groupBox_5);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        gridLayout_7->addWidget(label_9, 3, 0, 1, 1);

        label_13 = new QLabel(groupBox_5);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        gridLayout_7->addWidget(label_13, 5, 0, 1, 1);

        channelCountSpinBox = new QSpinBox(groupBox_5);
        channelCountSpinBox->setObjectName(QString::fromUtf8("channelCountSpinBox"));

        gridLayout_7->addWidget(channelCountSpinBox, 5, 2, 1, 1);


        gridLayout->addWidget(groupBox_5, 2, 0, 1, 1);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);

        gridLayout->addItem(verticalSpacer, 3, 0, 1, 1);

        groupBox_4 = new QGroupBox(ChangeSettings);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        gridLayout_6 = new QGridLayout(groupBox_4);
        gridLayout_6->setSpacing(6);
        gridLayout_6->setContentsMargins(11, 11, 11, 11);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        gridLayout_6->setHorizontalSpacing(6);
        btnBrowseOdas = new QPushButton(groupBox_4);
        btnBrowseOdas->setObjectName(QString::fromUtf8("btnBrowseOdas"));

        gridLayout_6->addWidget(btnBrowseOdas, 2, 2, 1, 1);

        label_6 = new QLabel(groupBox_4);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout_6->addWidget(label_6, 3, 0, 1, 1);

        label_2 = new QLabel(groupBox_4);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout_6->addWidget(label_2, 0, 0, 1, 1);

        odasPath = new QLineEdit(groupBox_4);
        odasPath->setObjectName(QString::fromUtf8("odasPath"));
        odasPath->setEnabled(false);
        sizePolicy.setHeightForWidth(odasPath->sizePolicy().hasHeightForWidth());
        odasPath->setSizePolicy(sizePolicy);
        odasPath->setMinimumSize(QSize(253, 0));
        odasPath->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        gridLayout_6->addWidget(odasPath, 2, 1, 1, 1);

        cameraConfigPath = new QLineEdit(groupBox_4);
        cameraConfigPath->setObjectName(QString::fromUtf8("cameraConfigPath"));
        cameraConfigPath->setEnabled(false);
        sizePolicy.setHeightForWidth(cameraConfigPath->sizePolicy().hasHeightForWidth());
        cameraConfigPath->setSizePolicy(sizePolicy);
        cameraConfigPath->setMinimumSize(QSize(540, 0));

        gridLayout_6->addWidget(cameraConfigPath, 0, 1, 1, 1);

        label = new QLabel(groupBox_4);
        label->setObjectName(QString::fromUtf8("label"));
        QSizePolicy sizePolicy1(QSizePolicy::Minimum, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy1);

        gridLayout_6->addWidget(label, 2, 0, 1, 1);

        micConfigPath = new QLineEdit(groupBox_4);
        micConfigPath->setObjectName(QString::fromUtf8("micConfigPath"));
        micConfigPath->setEnabled(false);
        sizePolicy.setHeightForWidth(micConfigPath->sizePolicy().hasHeightForWidth());
        micConfigPath->setSizePolicy(sizePolicy);
        micConfigPath->setMinimumSize(QSize(253, 0));

        gridLayout_6->addWidget(micConfigPath, 1, 1, 1, 1);

        label_3 = new QLabel(groupBox_4);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy2);
        label_3->setMinimumSize(QSize(0, 0));

        gridLayout_6->addWidget(label_3, 1, 0, 1, 1);

        btnBrowseMicConfig = new QPushButton(groupBox_4);
        btnBrowseMicConfig->setObjectName(QString::fromUtf8("btnBrowseMicConfig"));

        gridLayout_6->addWidget(btnBrowseMicConfig, 1, 2, 1, 1);

        btnBrowseCameraConfig = new QPushButton(groupBox_4);
        btnBrowseCameraConfig->setObjectName(QString::fromUtf8("btnBrowseCameraConfig"));

        gridLayout_6->addWidget(btnBrowseCameraConfig, 0, 2, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_6->addItem(horizontalSpacer, 1, 3, 1, 1);

        cbFaceDetection = new QComboBox(groupBox_4);
        cbFaceDetection->setObjectName(QString::fromUtf8("cbFaceDetection"));
        sizePolicy.setHeightForWidth(cbFaceDetection->sizePolicy().hasHeightForWidth());
        cbFaceDetection->setSizePolicy(sizePolicy);

        gridLayout_6->addWidget(cbFaceDetection, 3, 1, 1, 1);


        gridLayout->addWidget(groupBox_4, 1, 0, 1, 1);

        groupBox_3 = new QGroupBox(ChangeSettings);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        gridLayout_5 = new QGridLayout(groupBox_3);
        gridLayout_5->setSpacing(6);
        gridLayout_5->setContentsMargins(11, 11, 11, 11);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        defaultOutputFolder = new QLineEdit(groupBox_3);
        defaultOutputFolder->setObjectName(QString::fromUtf8("defaultOutputFolder"));
        defaultOutputFolder->setEnabled(false);
        sizePolicy.setHeightForWidth(defaultOutputFolder->sizePolicy().hasHeightForWidth());
        defaultOutputFolder->setSizePolicy(sizePolicy);
        defaultOutputFolder->setMinimumSize(QSize(540, 0));

        gridLayout_5->addWidget(defaultOutputFolder, 0, 1, 1, 1);

        btnBrowseDefaultOutputFolder = new QPushButton(groupBox_3);
        btnBrowseDefaultOutputFolder->setObjectName(QString::fromUtf8("btnBrowseDefaultOutputFolder"));

        gridLayout_5->addWidget(btnBrowseDefaultOutputFolder, 0, 2, 1, 1);

        label_5 = new QLabel(groupBox_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setMinimumSize(QSize(203, 0));

        gridLayout_5->addWidget(label_5, 0, 0, 1, 1);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_5->addItem(horizontalSpacer_3, 0, 3, 1, 1);


        gridLayout->addWidget(groupBox_3, 0, 0, 1, 1);


        retranslateUi(ChangeSettings);

        QMetaObject::connectSlotsByName(ChangeSettings);
    } // setupUi

    void retranslateUi(QWidget *ChangeSettings)
    {
        ChangeSettings->setWindowTitle(QApplication::translate("ChangeSettings", "Settings", nullptr));
        groupBox_5->setTitle(QApplication::translate("ChangeSettings", "Transcription", nullptr));
        label_4->setText(QApplication::translate("ChangeSettings", "Google service account", nullptr));
        enhancedCheckBox->setText(QString());
        label_7->setText(QApplication::translate("ChangeSettings", "Automatic transcription", nullptr));
        label_12->setText(QApplication::translate("ChangeSettings", "Use enhanced", nullptr));
        label_10->setText(QApplication::translate("ChangeSettings", "Model", nullptr));
        autoTranscriptionCheckBox->setText(QString());
        btnBrowseServiceAccount->setText(QApplication::translate("ChangeSettings", "Browse...", nullptr));
        label_8->setText(QApplication::translate("ChangeSettings", "Encoding", nullptr));
        label_11->setText(QApplication::translate("ChangeSettings", "Sample rate (Hz)", nullptr));
        label_9->setText(QApplication::translate("ChangeSettings", "Language", nullptr));
        label_13->setText(QApplication::translate("ChangeSettings", "Channel count", nullptr));
        groupBox_4->setTitle(QApplication::translate("ChangeSettings", "Conference", nullptr));
        btnBrowseOdas->setText(QApplication::translate("ChangeSettings", "Browse...", nullptr));
        label_6->setText(QApplication::translate("ChangeSettings", "Face detection method", nullptr));
        label_2->setText(QApplication::translate("ChangeSettings", "Camera configuration file", nullptr));
        odasPath->setText(QString());
        label->setText(QApplication::translate("ChangeSettings", "Odas library", nullptr));
        label_3->setText(QApplication::translate("ChangeSettings", "Microphone configuration file", nullptr));
        btnBrowseMicConfig->setText(QApplication::translate("ChangeSettings", "Browse...", nullptr));
        btnBrowseCameraConfig->setText(QApplication::translate("ChangeSettings", "Browse...", nullptr));
        groupBox_3->setTitle(QApplication::translate("ChangeSettings", "General", nullptr));
        btnBrowseDefaultOutputFolder->setText(QApplication::translate("ChangeSettings", "Browse...", nullptr));
        label_5->setText(QApplication::translate("ChangeSettings", "Default output folder", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ChangeSettings: public Ui_ChangeSettings {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CHANGE_SETTINGS_H
