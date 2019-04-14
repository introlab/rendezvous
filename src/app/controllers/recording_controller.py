from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.app.application_container import ApplicationContainer
from src.app.services.service.service_state import ServiceState
from src.app.services.recorder.recorder import Recorder


class RecordingController(QObject):

    signalException = pyqtSignal(Exception)
    signalRecordingState = pyqtSignal(bool)
    signalVirtualCamerasReceived = pyqtSignal(object)
    transcriptionReady = pyqtSignal()

    def __init__(self, parent=None):
        super(RecordingController, self).__init__(parent)

        self.__recorderState = ServiceState.TERMINATED

        self.__recorder = None
        self.__nChannels = 4
        self.__nChannelFile = 1
        self.__byteDepth = 2
        self.__sampleRate = 48000

        self.__caughtExceptions = []

        self.__odasServer = ApplicationContainer.odas()

        self.__videoProcessor = ApplicationContainer.videoProcessor()
    
        self.speechToText = ApplicationContainer.speechToText()
        
    def startRecording(self, outputFolder, odasPath, micConfigPath, cameraConfigPath, faceDetection):
        if self.__odasServer.state != ServiceState.STOPPED or self.__videoProcessor.state != ServiceState.STOPPED:
            self.signalRecordingState.emit(False)
            self.signalException.emit(Exception('Conference already started'))
            return

        if self.__recorderState == ServiceState.TERMINATED:
            self.__initializeRecorder(outputFolder)
            self.__startVideoProcessor(cameraConfigPath, faceDetection)
            self.__startOdasLive(odasPath, micConfigPath)


    def stopRecording(self):
        if self.__recorder and self.__recorderState == ServiceState.RUNNING:
            self.__recorder.stop()
            
        # TODO - call requestTranscription() with the audio file in input.

    
    def close(self):
        self.cancelTranscription()
        
        self.stopRecording()
        
        if self.__odasServer.isRunning:
            self.__odasServer.stop()


    @pyqtSlot(object)
    def __odasStateChanged(self, serviceState):
        self.__odasServer.state = serviceState

        if self.__odasServer.state == ServiceState.STARTING:
            self.__odasServer.signalException.connect(self.__exceptionHandling)

        elif self.__odasServer.state == ServiceState.RUNNING:
            self.__odasServer.signalAudioData.connect(self.__audioDataReceived)

        elif self.__odasServer.state == ServiceState.STOPPING:
            self.__odasServer.signalAudioData.disconnect(self.__audioDataReceived)

        elif self.__odasServer.state == ServiceState.STOPPED:
            self.__odasServer.signalException.disconnect(self.__exceptionHandling)
            self.__odasServer.signalStateChanged.disconnect(self.__odasStateChanged)


    @pyqtSlot(object)
    def __videoProcessorStateChanged(self, serviceState):
        self.__videoProcessor.state = serviceState

        if self.__videoProcessor.state == ServiceState.STARTING:
            self.__videoProcessor.signalException.connect(self.__exceptionHandling)
            self.__videoProcessor.signalVirtualCameras.connect(self.__virtualCamerasReceived)

        elif self.__videoProcessor.state == ServiceState.STOPPING:
            self.__videoProcessor.signalVirtualCameras.disconnect(self.__virtualCamerasReceived)

        elif self.__videoProcessor.state == ServiceState.STOPPED:
            self.__videoProcessor.signalException.disconnect(self.__exceptionHandling)
            self.__videoProcessor.signalStateChanged.disconnect(self.__videoProcessorStateChanged)


    @pyqtSlot(object)
    def __recorderStateChanged(self, serviceState):
        self.__recorderState = serviceState

        if self.__recorderState == ServiceState.STARTING:
            self.__recorder.signalException.connect(self.__exceptionHandling)

        elif self.__recorderState == ServiceState.READY:
            if self.__odasServer.state == ServiceState.RUNNING and self.__videoProcessor.state == ServiceState.RUNNING:
                self.__recorder.startRecording()

        elif self.__recorderState == ServiceState.RUNNING:
            self.signalRecordingState.emit(True)

        elif self.__recorderState == ServiceState.STOPPING:
            self.__stopVideoProcessor()
            self.__stopOdasLive()

        elif self.__recorderState == ServiceState.STOPPED:
            if self.__odasServer.state == ServiceState.STOPPED and self.__videoProcessor.state == ServiceState.STOPPED:
                self.__recorder.terminate()

        elif self.__recorderState == ServiceState.TERMINATED:
            self.__recorder.signalException.disconnect(self.__exceptionHandling)
            self.__recorder.signalStateChanged.disconnect(self.__recorderStateChanged)
            self.__recorder = None
            for e in self.__caughtExceptions:
                self.signalException.emit(e)
                self.__caughtExceptions.clear()
            self.signalRecordingState.emit(False)


    @pyqtSlot(bytes)
    def __audioDataReceived(self, streamData):
        if self.__recorder and self.__recorderState == ServiceState.RUNNING:
            self.__recorder.mailbox.put(('audio', streamData))


    @pyqtSlot(object, object)
    def __virtualCamerasReceived(self, images, virtualCameras):
        if self.__recorder and self.__recorderState == ServiceState.RUNNING:
            self.signalVirtualCamerasReceived.emit(images)

            if images:
                self.__recorder.mailbox.put(('video', images[0]))


    @pyqtSlot(Exception)
    def __exceptionHandling(self, e):
        self.__recorder.stop()
        self.__caughtExceptions.append(e)


    def __startVideoProcessor(self, cameraConfigPath, faceDetection):
        if self.__videoProcessor.state == ServiceState.STOPPED:
            self.__videoProcessor.signalStateChanged.connect(self.__videoProcessorStateChanged)
            self.__videoProcessor.start(cameraConfigPath, faceDetection)
        

    def __stopVideoProcessor(self):
        self.__videoProcessor.stop()


    def __startOdasLive(self, odasPath, micConfigPath):
        self.__odasServer.signalStateChanged.connect(self.__odasStateChanged)
        self.__odasServer.startOdasLive(odasPath, micConfigPath)  


    def __initializeRecorder(self, outputFolder):
        self.__recorder = Recorder(outputFolder, self.__nChannels, self.__nChannelFile, self.__byteDepth, self.__sampleRate)
        self.__recorder.signalStateChanged.connect(self.__recorderStateChanged)
        self.__recorder.initialize()


    def __stopOdasLive(self):
        self.__odasServer.stopOdasLive()
        
        
    def requestTranscription(self, audioDataPath):
        if not self.speechToText.isRunning:
            # We need to set the config since we can't pass arguments directly to the thread.
            settings = ApplicationContainer.settings()
            config = {
                'audioDataPath' : audioDataPath,
                'encoding' : settings.getValue('speechToTextEncoding'),
                'enhanced' : bool(settings.getValue('speechToTextEnhanced')),
                'languageCode' : settings.getValue('speechToTextLanguage'),
                'model' : settings.getValue('speechToTextModel'),
                'outputFolder' : settings.getValue('defaultOutputFolder'),
                'sampleRate' : int(settings.getValue('speechToTextSampleRate')),
                'audioChannelCount' : int(settings.getValue('speechToTextChannelCount')),
                'serviceAccountPath' : settings.getValue('serviceAccountPath')
            }
            self.speechToText.setConfig(config)
            self.connectSignals()
            self.speechToText.asynchroneSpeechToText.start()
        
        else:
            self.signalException.emit(Exception('Transcription is already running'))


    def cancelTranscription(self):
        if self.speechToText.isRunning:
            self.speechToText.asynchroneSpeechToText.quit()
            self.disconnectSignals()


    def connectSignals(self):
        self.speechToText.transcriptionReady.connect(self.onTranscriptionReady)
        self.speechToText.exception.connect(self.onTranscriptionException)


    def disconnectSignals(self):
        self.speechToText.transcriptionReady.disconnect(self.onTranscriptionReady)
        self.speechToText.exception.disconnect(self.onTranscriptionException)


    @pyqtSlot()
    def onTranscriptionReady(self):
        self.transcriptionReady.emit()
        self.speechToText.asynchroneSpeechToText.quit()
        self.disconnectSignals()


    @pyqtSlot(Exception)
    def onTranscriptionException(self, e):
        self.signalException.emit(e)
        self.speechToText.asynchroneSpeechToText.quit()
        self.disconnectSignals()


