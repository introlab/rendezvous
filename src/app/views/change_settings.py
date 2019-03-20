import os
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot

from src.app.gui.change_settings_ui import Ui_ChangeSettings


class ChangeSettings(QWidget, Ui_ChangeSettings):

    rootDirectory = str(Path(__file__).resolve().parents[4])

    def __init__(self, parent=None):
        super(ChangeSettings, self).__init__(parent)
        self.setupUi(self)

        self.__loadSettings(parent)

        self.btnBrowseCameraConfig.clicked.connect(self.btnBrowseCameraConfigClicked)
        self.btnBrowseServiceAccount.clicked.connect(self.btnBrowseServiceAccountClicked)
        self.btnBrowseMicConfig.clicked.connect(self.btnBrowseMicConfigClicked)
        self.btnBrowseOdas.clicked.connect(self.btnBrowseOdasClicked)
        self.btnBrowseDefaultOutputFolder.clicked.connect(self.btnBrowseDefaultOutputFolderClicked)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            event.accept()


    @pyqtSlot()
    def btnBrowseOdasClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        odasPath, _ = QFileDialog.getOpenFileName(self, 'Browse Odas Path', self.rootDirectory, options=options)
        if odasPath:
            self.odasPath.setText(odasPath)
            self.window().setSetting('odasPath', odasPath)
        

    @pyqtSlot()
    def btnBrowseMicConfigClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        micConfigPath, _ = QFileDialog.getOpenFileName(self, 'Browse Microphone Config', self.rootDirectory, 'Config Files (*.cfg)', options=options)
        if micConfigPath:
            self.micConfigPath.setText(micConfigPath)
            self.window().setSetting('micConfigPath', micConfigPath)


    @pyqtSlot()
    def btnBrowseCameraConfigClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        cameraConfigPath, _ = QFileDialog.getOpenFileName(self, 'Browse Camera Config', self.rootDirectory, 'JSON Files (*.json)', options=options)
        if cameraConfigPath:
            self.cameraConfigPath.setText(cameraConfigPath)
            self.window().setSetting('cameraConfigPath', cameraConfigPath)


    @pyqtSlot()
    def btnBrowseServiceAccountClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        serviceAccountPath, _ = QFileDialog.getOpenFileName(self, 'Browse Google Service Account File', self.rootDirectory, 'JSON Files (*.json)', options=options)
        if serviceAccountPath:
            self.serviceAccountPath.setText(serviceAccountPath)
            self.window().setSetting('serviceAccountPath', serviceAccountPath)


    @pyqtSlot()
    def btnBrowseDefaultOutputFolderClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        defaultOutputFolder = QFileDialog.getExistingDirectory(self, 'Browse Default Output Folder', self.rootDirectory, options=options)
        if defaultOutputFolder:
            self.defaultOutputFolder.setText(defaultOutputFolder)
            self.window().setSetting('defaultOutputFolder', defaultOutputFolder)


    def __loadSettings(self, parent):
        self.cameraConfigPath.setText(str(parent.getSetting('cameraConfigPath')))
        self.serviceAccountPath.setText(str(parent.getSetting('serviceAccountPath')))
        self.micConfigPath.setText(str(parent.getSetting('micConfigPath')))
        self.odasPath.setText(str(parent.getSetting('odasPath')))
        self.defaultOutputFolder.setText(str(parent.getSetting('defaultOutputFolder')))
        