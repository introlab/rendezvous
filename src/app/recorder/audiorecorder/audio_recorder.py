import wave
import os
from threading import Thread
from time import sleep
import socket

from PyQt5.QtCore import QObject, pyqtSignal

class AudioRecorder(QObject):

    signalException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(AudioRecorder, self).__init__(parent)
        self.isRunning = False
        self.isRecording = False
        self.isConnected = False
        self.isOdasClosed = False
        self.connection = None
        self.wavFileWriter = None
        self.audioBuffer = bytearray()
    
    def startServer(self):
        try:
            Thread(target=self.__run, args=[]).start()
        
        except Exception as e:
            self.isRunning = False
            self.signalException.emit(e)
    

    def stopServer(self):
        if self.isRecording:
            self.stopRecording()
        self.closeConnection()
        self.isRunning = False

    
    def closeConnection(self, isFromUI=False):
        if self.connection:
            self.connection.close()
            self.isConnected = False
            self.connection = None
            print('connection closed')

        if isFromUI:
            self.isOdasClosed = True
    
    def startRecording(self, outputFolder):
        outputFile = os.path.join(outputFolder, 'outputAudio.wav')
        self.wavFileWriter = wave.open(outputFile, 'wb')
        self.wavFileWriter.setnchannels(4)
        self.wavFileWriter.setsampwidth(2)
        self.wavFileWriter.setframerate(48000)
        self.isRecording = True


    def stopRecording(self):
        if self.wavFileWriter:
            self.wavFileWriter.writeframesraw(self.audioBuffer)
            self.audioBuffer = bytearray()
            self.wavFileWriter.close()
            self.wavFileWriter = None
        self.isRecording = False


    def __run(self):
        try:
            host = '127.0.0.1'
            port = 10020

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.sock:
                self.sock.bind((host, port))
                self.sock.listen()
                print('server is up!')
                self.isRunning = True
                
                while True:
                    if not self.isRunning:
                        break
                    
                    if not self.isOdasClosed:
                        self.connection, _ = self.sock.accept()
                        if self.connection:
                            self.isConnected = True
                            print('client connected!')
                            while True:
                                if not self.isConnected or not self.isRunning:
                                    break

                                if self.isRecording:
                                    data = self.connection.recv(1024)

                                    if not data:
                                        break

                                    self.audioBuffer += data
                                sleep(0.00001)
                    
                    sleep(0.00001)

        except Exception as e:
            self.signalException.emit(e)

        finally:
            self.stopServer()
            print('server stopped')
