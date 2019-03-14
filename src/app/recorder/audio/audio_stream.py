import os
from threading import Thread
from time import sleep
import socket

from PyQt5.QtCore import QObject, pyqtSignal

class AudioStream(QObject):

    signalException = pyqtSignal(Exception)
    signalServerUp = pyqtSignal()
    signalServerDown = pyqtSignal()
    signalNewData = pyqtSignal(bytes)

    def __init__(self, hostIP, port, parent=None):
        super(AudioStream, self).__init__(parent)
        
        self.host = hostIP
        self.port = port

        self.isRunning = False
        self.isConnected = False
        self.isOdasClosed = False
        self.isVerbose = False

        self.clientConnection = None

    
    def startServer(self, isVerbose=False):
        try:
            self.isVerbose = isVerbose
            Thread(target=self.__run, args=[]).start()
        
        except Exception as e:
            self.isRunning = False
            self.signalServerDown.emit()
            self.signalException.emit(e)
    

    def stopServer(self):
        self.closeConnection()
        self.isRunning = False
        self.signalServerDown.emit()

    
    def closeConnection(self, isFromUI=False):
        if self.clientConnection:
            self.clientConnection.close()
            self.isConnected = False
            self.clientConnection = None
            print('connection closed') if self.isVerbose else None

        if isFromUI:
            self.isOdasClosed = True
    

    def __run(self):
        try:

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self.sock:
                self.sock.bind((self.hostIP, self.port))
                self.sock.listen()
                self.isRunning = True
                self.signalServerUp.emit()
                print('server is up!') if self.isVerbose else None

                while True:
                    if not self.isRunning:
                        break
                    
                    if not self.isOdasClosed:
                        self.clientConnection, _ = self.sock.accept()
                        if self.clientConnection:
                            self.isConnected = True
                            print('client connected!') if self.isVerbose else None
                            while True:
                                if not self.isConnected or not self.isRunning:
                                    break

                                # 1024 because this is the minimum Odas send through the socket.
                                data = self.clientConnection.recv(1024)
                                # if there is no data incomming close the stream.
                                if not data:
                                    break

                                self.signalNewData(data)
                                sleep(0.00001)
                    
                    sleep(0.00001)

        except Exception as e:
            self.signalException.emit(e)

        finally:
            self.stopServer()
            print('server stopped') if self.isVerbose else None
