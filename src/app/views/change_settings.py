import os
from pathlib import Path

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QWidget

from src.app.application_container import ApplicationContainer
from src.app.gui.change_settings_ui import Ui_ChangeSettings
from src.app.services.speechtotext.speech_to_text import EncodingTypes, LanguageCodes, Models
from src.app.services.videoprocessing.facedetection.facedetector.face_detection_methods import FaceDetectionMethods


class ChangeSettings(QWidget, Ui_ChangeSettings):

    def __init__(self, parent=None):
        super(ChangeSettings, self).__init__(parent)
        self.setupUi(self)

        # Populate UI.
        self.cbFaceDetection.addItems([faceDetectionMethod.value for faceDetectionMethod in FaceDetectionMethods])
        self.encodingComboBox.addItems([encodingType.value for encodingType in EncodingTypes])
        self.languageComboBox.addItems([languageCode.value for languageCode in LanguageCodes])
        self.modelComboBox.addItems([model.value for model in Models])
        self.sampleRateSpinBox.setRange(ApplicationContainer.speechToText().getMinSampleRate(), ApplicationContainer.speechToText().getMaxSampleRate())
 
        # Load settings.
        self.cameraConfigPath.setText(ApplicationContainer.settings().getValue('cameraConfigPath'))
        self.micConfigPath.setText(ApplicationContainer.settings().getValue('micConfigPath'))
        self.odasPath.setText(ApplicationContainer.settings().getValue('odasPath'))
        self.defaultOutputFolder.setText(ApplicationContainer.settings().getValue('defaultOutputFolder'))

        self.cbFaceDetection.setCurrentIndex(self.cbFaceDetection.findText(ApplicationContainer.settings().getValue('faceDetection')))

        self.serviceAccountPath.setText(ApplicationContainer.settings().getValue('serviceAccountPath'))
        self.encodingComboBox.setCurrentIndex(self.encodingComboBox.findText(ApplicationContainer.settings().getValue('speechToTextEncoding')))
        self.languageComboBox.setCurrentIndex(self.languageComboBox.findText(ApplicationContainer.settings().getValue('speechToTextLanguage')))  
        self.modelComboBox.setCurrentIndex(self.modelComboBox.findText(ApplicationContainer.settings().getValue('speechToTextModel')))
        self.sampleRateSpinBox.setValue(int(ApplicationContainer.settings().getValue('speechToTextSampleRate')))
        self.autoTranscriptionCheckBox.setCheckState(2 if int(ApplicationContainer.settings().getValue('automaticTranscription')) else 0)
        self.enhancedCheckBox.setCheckState(2 if int(ApplicationContainer.settings().getValue('speechToTextEnhanced')) else 0)

        # Qt signal slots.
        self.btnBrowseCameraConfig.clicked.connect(self.btnBrowseCameraConfigClicked)
        self.btnBrowseServiceAccount.clicked.connect(self.btnBrowseServiceAccountClicked)
        self.btnBrowseMicConfig.clicked.connect(self.btnBrowseMicConfigClicked)
        self.btnBrowseOdas.clicked.connect(self.btnBrowseOdasClicked)
        self.btnBrowseDefaultOutputFolder.clicked.connect(self.btnBrowseDefaultOutputFolderClicked)
        self.cbFaceDetection.currentIndexChanged.connect(self.cbFaceDetectionIndexChanged)

        self.encodingComboBox.currentIndexChanged.connect(self.encodingComboBoxChanged)
        self.languageComboBox.currentIndexChanged.connect(self.languageComboBoxChanged)
        self.modelComboBox.currentIndexChanged.connect(self.modelComboBoxChanged)
        self.sampleRateSpinBox.valueChanged.connect(self.sampleRateSpinBoxChanged)
        self.autoTranscriptionCheckBox.stateChanged.connect(self.autoTranscriptionCheckBoxChanged)
        self.enhancedCheckBox.stateChanged.connect(self.enhancedCheckBoxChanged)


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


    @pyqtSlot()
    def encodingComboBoxChanged(self):
        ApplicationContainer.settings().setValue('speechToTextEncoding', self.encodingComboBox.currentText())


    @pyqtSlot()
    def languageComboBoxChanged(self):
        ApplicationContainer.settings().setValue('speechToTextLanguage', self.languageComboBox.currentText())


    @pyqtSlot()
    def modelComboBoxChanged(self):
        ApplicationContainer.settings().setValue('speechToTextModel', self.modelComboBox.currentText())


    @pyqtSlot(int)
    def sampleRateSpinBoxChanged(self, value):
        ApplicationContainer.settings().setValue('speechToTextSampleRate', self.sampleRateSpinBox.value())


    @pyqtSlot(int)
    def autoTranscriptionCheckBoxChanged(self, state):
        value = 1 if state == 2 else 0
        ApplicationContainer.settings().setValue('automaticTranscription', value)


    @pyqtSlot(int)
    def enhancedCheckBoxChanged(self, state):
        value = 1 if state == 2 else 0
        ApplicationContainer.settings().setValue('speechToTextEnhanced', value)
        