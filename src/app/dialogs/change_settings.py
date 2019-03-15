import os
from pathlib import Path

from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtCore import pyqtSlot

from src.app.gui.change_settings_ui import Ui_ChangeSettings


class ChangeSettings(QDialog, Ui_ChangeSettings):

    rootDirectory = os.path.realpath(Path(__file__).parents[4])

    def __init__(self, parent=None):
        super(ChangeSettings, self).__init__(parent)
        self.setupUi(self)

        self.__loadSettings(parent)

        self.dialogBtnBox.accepted.connect(self.apply)
        self.dialogBtnBox.rejected.connect(self.cancel)

        self.btnBrowseCameraConfig.clicked.connect(self.btnBrowseCameraConfigClicked)
        self.btnBrowseServiceAccount.clicked.connect(self.btnBrowseServiceAccountClicked)
        self.btnBrowseMicConfig.clicked.connect(self.btnBrowseMicConfigClicked)
        self.btnBrowseOdas.clicked.connect(self.btnBrowseOdasClicked)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            event.accept()


    @pyqtSlot()
    def apply(self):
        self.parent().setSetting('cameraConfigPath', self.cameraConfigPath.text())
        self.parent().setSetting('serviceAccountPath', self.serviceAccountPath.text())
        self.parent().setSetting('micConfigPath', self.micConfigPath.text())
        self.parent().setSetting('odasPath', self.odasPath.text())
        
        self.accept()


    @pyqtSlot()
    def cancel(self):
        self.reject()


    @pyqtSlot()
    def btnBrowseOdasClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        odasPath, _ = QFileDialog.getOpenFileName(self, 'Browse Odas Path', self.rootDirectory, options=options)
        if odasPath:
            self.odasPath.setText(odasPath)
        

    @pyqtSlot()
    def btnBrowseMicConfigClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        micConfigPath, _ = QFileDialog.getOpenFileName(self, 'Browse Microphone Config', self.rootDirectory, 'Config Files (*.cfg)', options=options)
        if micConfigPath:
            self.micConfigPath.setText(micConfigPath)


    @pyqtSlot()
    def btnBrowseCameraConfigClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        cameraConfigPath, _ = QFileDialog.getOpenFileName(self, 'Browse Camera Config', self.rootDirectory, 'JSON Files (*.json)', options=options)
        if cameraConfigPath:
            self.cameraConfigPath.setText(cameraConfigPath)

    @pyqtSlot()
    def btnBrowseServiceAccountClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        serviceAccountPath, _ = QFileDialog.getOpenFileName(self, 'Browse Google Service Account File', self.rootDirectory, 'JSON Files (*.json)', options=options)
        if serviceAccountPath:
            self.serviceAccountPath.setText(serviceAccountPath)


    def __loadSettings(self, parent):
        self.cameraConfigPath.setText(parent.getSetting('cameraConfigPath'))
        self.serviceAccountPath.setText(parent.getSetting('serviceAccountPath'))
        self.micConfigPath.setText(parent.getSetting('micConfigPath'))
        self.odasPath.setText(parent.getSetting('odasPath'))
        