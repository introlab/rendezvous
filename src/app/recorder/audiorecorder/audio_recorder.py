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
        self.wavFileWriters = []
        self.sourcesBuffer = {'0' :bytearray(), '1': bytearray(), '2': bytearray(), '3': bytearray()}
    
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
        for i in range(0, 4):
            outputFile = os.path.join(outputFolder, 'outputsrc-{}.wav'.format(i))
            self.wavFileWriters.append(wave.open(outputFile, 'wb'))
            self.wavFileWriters[i].setnchannels(1)
            self.wavFileWriters[i].setsampwidth(2)
            self.wavFileWriters[i].setframerate(48000)
        
        self.isRecording = True


    def stopRecording(self):
        for index, wavWriter in enumerate(self.wavFileWriters):
            if wavWriter:
                audioRaw = self.sourcesBuffer[str(index)]
                wavWriter.writeframesraw(audioRaw)
                wavWriter.close()
                self.sourcesBuffer[str(index)] = bytearray()
        
        self.wavFileWriters = []
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

                                data = self.connection.recv(1024)
                                if not data:
                                    break

                                if self.isRecording:
                                    nChannel = 4
                                    offset = 0
                                    while offset < len(data):
                                        for key, _ in self.sourcesBuffer.items():
                                            currentByte = int(offset + int(key))
                                            self.sourcesBuffer[key] += data[currentByte:currentByte + 2]

                                        offset += nChannel * 2
                                        

                                sleep(0.00001)
                    
                    sleep(0.00001)

        except Exception as e:
            self.signalException.emit(e)
            raise(e)

        finally:
            self.stopServer()
            print('server stopped')
