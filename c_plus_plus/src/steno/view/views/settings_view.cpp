#include "settings_view.h"
#include "ui_settings_view.h"

#include <iostream>
#include <QString>
#include <QVariant>

using namespace std;



namespace View
{
    SettingsView::SettingsView(QWidget *parent)
        : AbstractView("Settings", parent)
        , m_ui(new Ui::SettingsView)
        , m_dialog(this)
        , m_structAppSettings(new View::ApplicationSettingsStruct)
    {
        //Buttons
        m_ui->setupUi(this);
        connect(m_ui->btnBrowseDefaultOutputFolder, &QAbstractButton::clicked, [=]{m_dialog.open();});
        connect(m_ui->btnBrowseCameraConfig, &QAbstractButton::clicked, [=]{m_dialog.open();});
        connect(m_ui->btnBrowseMicConfig, &QAbstractButton::clicked, [=]{m_dialog.open();});
        connect(m_ui->btnBrowseOdas, &QAbstractButton::clicked, [=]{m_dialog.open();});
        connect(m_ui->btnBrowseServiceAccount, &QAbstractButton::clicked, [=]{m_dialog.open();});

        // General Group Box Init
        m_ui->txtBoxDefaultOutputFolder->setText(QString::fromStdString(m_structAppSettings->general.defaultConfigurationFilePath));

        // Conference Group Box Init
        m_ui->txtBoxCameraConfigPath->setText(QString::fromStdString(m_structAppSettings->conference.cameraConfigurationFilePath));
        m_ui->txtBoxMicConfigPath->setText(QString::fromStdString(m_structAppSettings->conference.microConfigurationFilePath));
        m_ui->txtBoxOdasPath->setText(QString::fromStdString(m_structAppSettings->conference.odasLibraryFilePath));
        //std::string s = m_structAppSettings->transcription.AMR.toString();
        //m_ui->comboBoxFaceDetection->addItems(QVariant::fromValue(m_structAppSettings->conference.YOLOV3).toString());
    }

    SettingsView::~SettingsView()
    {
        delete m_ui;
    }

} // View
