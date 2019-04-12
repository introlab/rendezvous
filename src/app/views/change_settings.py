import os
from pathlib import Path

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QWidget

from src.app.application_container import ApplicationContainer
from src.app.gui.change_settings_ui import Ui_ChangeSettings
from src.app.services.videoprocessing.facedetection.facedetector.face_detection_methods import FaceDetectionMethods


class ChangeSettings(QWidget, Ui_ChangeSettings):

    def __init__(self, parent=None):
        super(ChangeSettings, self).__init__(parent)
        self.setupUi(self)

        self.__loadSettings()

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
            ApplicationContainer.settings().setValue('odasPath', odasPath)
        

    @pyqtSlot()
    def btnBrowseMicConfigClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        micConfigPath, _ = QFileDialog.getOpenFileName(self, 'Browse Microphone Config', self.window().rootDirectory, 'Config Files (*.cfg)', options=options)
        if micConfigPath:
            self.micConfigPath.setText(micConfigPath)
            ApplicationContainer.settings().setValue('micConfigPath', micConfigPath)


    @pyqtSlot()
    def btnBrowseCameraConfigClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        cameraConfigPath, _ = QFileDialog.getOpenFileName(self, 'Browse Camera Config', self.window().rootDirectory, 'JSON Files (*.json)', options=options)
        if cameraConfigPath:
            self.cameraConfigPath.setText(cameraConfigPath)
            ApplicationContainer.settings().setValue('cameraConfigPath', cameraConfigPath)


    @pyqtSlot()
    def btnBrowseServiceAccountClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        serviceAccountPath, _ = QFileDialog.getOpenFileName(self, 'Browse Google Service Account File', self.window().rootDirectory, 'JSON Files (*.json)', options=options)
        if serviceAccountPath:
            self.serviceAccountPath.setText(serviceAccountPath)
            ApplicationContainer.settings().setValue('serviceAccountPath', serviceAccountPath)


    @pyqtSlot()
    def btnBrowseDefaultOutputFolderClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        defaultOutputFolder = QFileDialog.getExistingDirectory(self, 'Browse Default Output Folder', self.window().rootDirectory, options=options)
        if defaultOutputFolder:
            self.defaultOutputFolder.setText(defaultOutputFolder)
            ApplicationContainer.settings().setValue('defaultOutputFolder', defaultOutputFolder)


    @pyqtSlot()
    def cbFaceDetectionIndexChanged(self):
        ApplicationContainer.settings().setValue('faceDetection', self.cbFaceDetection.currentText())


    def __loadSettings(self):
        self.cameraConfigPath.setText(ApplicationContainer.settings().getValue('cameraConfigPath'))
        self.serviceAccountPath.setText(ApplicationContainer.settings().getValue('serviceAccountPath'))
        self.micConfigPath.setText(ApplicationContainer.settings().getValue('micConfigPath'))
        self.odasPath.setText(ApplicationContainer.settings().getValue('odasPath'))
        self.defaultOutputFolder.setText(ApplicationContainer.settings().getValue('defaultOutputFolder'))

        self.cbFaceDetection.addItems([faceDetectionMethod.value for faceDetectionMethod in FaceDetectionMethods])
        self.cbFaceDetection.setCurrentIndex(self.cbFaceDetection.findText(ApplicationContainer.settings().getValue('faceDetection')))
        
        