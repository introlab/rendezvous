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

        self.__loadSettings(self.parent().settingsManager)

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
        self.parent().settingsManager.setValue('cameraConfigPath', self.cameraConfigPath.text())
        self.parent().settingsManager.setValue('serviceAccountPath', self.serviceAccountPath.text())
        self.parent().settingsManager.setValue('micConfigPath', self.micConfigPath.text())
        self.parent().settingsManager.setValue('odasPath', self.odasPath.text())
        
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


    def __loadSettings(self, settingsManager):
        self.cameraConfigPath.setText(settingsManager.getValue('cameraConfigPath'))
        self.serviceAccountPath.setText(settingsManager.getValue('serviceAccountPath'))
        self.micConfigPath.setText(settingsManager.getValue('micConfigPath'))
        self.odasPath.setText(settingsManager.getValue('odasPath'))
        