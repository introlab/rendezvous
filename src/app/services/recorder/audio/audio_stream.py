import os
from threading import Thread
from time import sleep
import socket

from PyQt5.QtCore import QObject, pyqtSignal

class AudioStream(QObject, Thread):

    signalException = pyqtSignal(Exception)
    signalServerUp = pyqtSignal()
    signalServerDown = pyqtSignal()
    signalNewData = pyqtSignal(bytes)

    def __init__(self, hostIP, port, isVerbose=False, parent=None):
        super(AudioStream, self).__init__(parent)
        Thread.__init__(self)

        self.daemon = True
        
        self.host = hostIP
        self.port = port
        self.isVerbose = isVerbose

        self.isRunning = False
        self.isConnected = False

        self.clientConnection = None
    

    def stop(self):
        self.closeConnection()
        self.isRunning = False
        self.signalServerDown.emit()
        print('server stopped') if self.isVerbose else None

    
    def closeConnection(self):
        if self.clientConnection:
            self.clientConnection.close()
            self.isConnected = False
            self.clientConnection = None
            print('connection closed') if self.isVerbose else None
    

    def run(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((self.host, self.port))
                sock.listen()
                self.isRunning = True
                self.signalServerUp.emit()
                print('server is up!') if self.isVerbose else None

                while True:
                    self.clientConnection, _ = sock.accept()
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
                            self.signalNewData.emit(data)
                            sleep(0.00001)
                    
                    sleep(0.00001)

        except Exception as e:
            self.signalException.emit(e)

