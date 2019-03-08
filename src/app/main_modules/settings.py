from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtCore import pyqtSlot

from src.app.gui.settings_ui import Ui_Settings
from src.app.settings.app_settings import AppSettings


class SettingsDialog(QDialog, Ui_Settings):
    
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)
        self.setupUi(self)

        self.__loadSettings()

        self.dialogBtnBox.accepted.connect(self.apply)
        self.dialogBtnBox.rejected.connect(self.cancel)

        self.btnBrowseOdas.clicked.connect(self.btnBrowseOdasClicked)
        self.btnBrowseMicConfig.clicked.connect(self.btnBrowseMicConfigClicked)
        self.btnBrowseCameraConfig.clicked.connect(self.btnBrowseCameraConfigClicked)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        event.accept()


    @pyqtSlot()
    def apply(self):
        AppSettings.setValue("odasPath", self.odasPath.text())
        AppSettings.setValue("micConfigPath", self.micConfigPath.text())
        AppSettings.setValue("cameraConfigPath", self.cameraConfigPath.text())
        self.accept()


    @pyqtSlot()
    def cancel(self):
        self.reject()


    @pyqtSlot()
    def btnBrowseOdasClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        odasPath, _ = QFileDialog.getOpenFileName(self, "Browse Odas Path", "./", options=options)
        if odasPath:
            self.odasPath.setText(odasPath)
        

    @pyqtSlot()
    def btnBrowseMicConfigClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        micConfigPath, _ = QFileDialog.getOpenFileName(self, "Browse Microphone Config", "./", "Config Files (*.cfg)", options=options)
        if micConfigPath:
            self.micConfigPath.setText(micConfigPath)


    @pyqtSlot()
    def btnBrowseCameraConfigClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        cameraConfigPath, _ = QFileDialog.getOpenFileName(self, "Browse Camera Config", "./", "JSON Files (*.json)", options=options)
        if cameraConfigPath:
            self.cameraConfigPath.setText(cameraConfigPath)


    def __loadSettings(self):
        self.odasPath.setText(AppSettings.getValue("odasPath"))
        self.micConfigPath.setText(AppSettings.getValue("micConfigPath"))
        self.cameraConfigPath.setText(AppSettings.getValue("cameraConfigPath"))
        