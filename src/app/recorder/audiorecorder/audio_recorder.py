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
        self.connection = None
        self.bitNumber = 16
        self.wavFileWriter = None
        self.audioBuffer = bytearray()
    
    def startServer(self):
        try:
            Thread(target=self.__run, args=[]).start()
        
        except Exception as e:
            self.isRunning = False
            self.signalException.emit(e)
    

    def stopServer(self):
        self.stopRecording()
        self.closeConnection()
        self.isRunning = False

    
    def closeConnection(self):
        if self.connection:
            self.connection.close()
            self.isConnected = False
            print('connection closed')
    
    def startRecording(self, outputFolder):
        outputFile = os.path.join(outputFolder, 'outputAudio.wav')
        self.wavFileWriter = wave.open(outputFile, 'wb')
        self.wavFileWriter.setnchannels(1)
        self.wavFileWriter.setsampwidth(2)
        self.wavFileWriter.setframerate(48000)
        self.isRecording = True


    def stopRecording(self):
        if self.audioBuffer:
            self.wavFileWriter.writeframesraw(self.audioBuffer)
            self.audioBuffer = bytearray()
            self.wavFileWriter.close()
        self.isRecording = False


    def __run(self):
        try:
            host = '127.0.0.1'
            port = 10020

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
                sock.listen()
                print('server is up!')
                self.isRunning = True
                
                while True:
                    if not self.isRunning:
                        break

                    self.connection, _ = sock.accept()
                    if self.connection:
                        self.isConnected = True
                        print('client connected!')
                        while True:
                            if not self.isConnected:
                                break

                            if self.isRecording:
                                data = self.connection.recv(1024)

                                if not data:
                                    break
                                print('I received data!! :)')
                                self.audioBuffer.extend(data)
                            sleep(0.00001)
                    
                    sleep(0.00001)

        except Exception as e:
            self.signalException.emit(e)
            raise e

        finally:
            self.stopServer()
            print('server stopped')

if __name__ == '__main__':
    recorder = AudioRecorder()
    recorder.start('/home/morel/development/rendezvous/output')
    sleep(3)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('127.0.0.1', 10020))

    client.send(bytes('bonsoir mr le serveur', 'utf-8'))
    sleep(2)
    client.send(bytes('BON MATIN mr le serveur', 'utf-8'))
    client.close()