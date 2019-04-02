import os
from pathlib import Path

from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import pyqtSlot

from src.app.gui.change_settings_ui import Ui_ChangeSettings
from src.app.services.videoprocessing.facedetection.facedetector.face_detection_methods import FaceDetectionMethods


class ChangeSettings(QWidget, Ui_ChangeSettings):

    def __init__(self, parent=None):
        super(ChangeSettings, self).__init__(parent)
        self.setupUi(self)

        self.__loadSettings(parent)

        self.btnBrowseCameraConfig.clicked.connect(self.btnBrowseCameraConfigClicked)
        self.btnBrowseServiceAccount.clicked.connect(self.btnBrowseServiceAccountClicked)
        self.btnBrowseMicConfig.clicked.connect(self.btnBrowseMicConfigClicked)
        self.btnBrowseOdas.clicked.connect(self.btnBrowseOdasClicked)
        self.btnBrowseDefaultOutputFolder.clicked.connect(self.btnBrowseDefaultOutputFolderClicked)
        self.cbFaceDetection.currentIndexChanged.connect(self.cbFaceDetectionIndexChanged)


    # Handles the event where the user closes the window with the X button
    def closeEvent(self, event):
        if event:
            event.accept()


    @pyqtSlot()
    def btnBrowseOdasClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        odasPath, _ = QFileDialog.getOpenFileName(self, 'Browse Odas Path', self.window().rootDirectory, options=options)
        if odasPath:
            self.odasPath.setText(odasPath)
            self.window().setSetting('odasPath', odasPath)
        

    @pyqtSlot()
    def btnBrowseMicConfigClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        micConfigPath, _ = QFileDialog.getOpenFileName(self, 'Browse Microphone Config', self.window().rootDirectory, 'Config Files (*.cfg)', options=options)
        if micConfigPath:
            self.micConfigPath.setText(micConfigPath)
            self.window().setSetting('micConfigPath', micConfigPath)


    @pyqtSlot()
    def btnBrowseCameraConfigClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        cameraConfigPath, _ = QFileDialog.getOpenFileName(self, 'Browse Camera Config', self.window().rootDirectory, 'JSON Files (*.json)', options=options)
        if cameraConfigPath:
            self.cameraConfigPath.setText(cameraConfigPath)
            self.window().setSetting('cameraConfigPath', cameraConfigPath)


    @pyqtSlot()
    def btnBrowseServiceAccountClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        serviceAccountPath, _ = QFileDialog.getOpenFileName(self, 'Browse Google Service Account File', self.window().rootDirectory, 'JSON Files (*.json)', options=options)
        if serviceAccountPath:
            self.serviceAccountPath.setText(serviceAccountPath)
            self.window().setSetting('serviceAccountPath', serviceAccountPath)


    @pyqtSlot()
    def btnBrowseDefaultOutputFolderClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        defaultOutputFolder = QFileDialog.getExistingDirectory(self, 'Browse Default Output Folder', self.window().rootDirectory, options=options)
        if defaultOutputFolder:
            self.defaultOutputFolder.setText(defaultOutputFolder)
            self.window().setSetting('defaultOutputFolder', defaultOutputFolder)


    @pyqtSlot()
    def cbFaceDetectionIndexChanged(self):
        self.window().setSetting('faceDetection', self.cbFaceDetection.currentText())


    def __loadSettings(self, parent):
        self.cameraConfigPath.setText(parent.getSetting('cameraConfigPath'))
        self.serviceAccountPath.setText(parent.getSetting('serviceAccountPath'))
        self.micConfigPath.setText(parent.getSetting('micConfigPath'))
        self.odasPath.setText(parent.getSetting('odasPath'))
        self.defaultOutputFolder.setText(parent.getSetting('defaultOutputFolder'))

        self.cbFaceDetection.addItems([faceDetectionMethod.value for faceDetectionMethod in FaceDetectionMethods])
        self.cbFaceDetection.setCurrentIndex(self.cbFaceDetection.findText(parent.getSetting('faceDetection')))
        
        