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
    
    
    def start(self, outputFolder):
        try:
            Thread(target=self.__run, args=[outputFolder]).start()
        
        except Exception as e:
            self.isRunning = False
            self.signalException.emit(e)
    

    def stop(self):
        self.isRunning = False


    def __run(self, outputFolder):
        try:
            host = '127.0.0.1'
            port = 10020

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
                server.bind((host, port))
                server.listen()
                print('server is up!')
                client, _ = server.accept()
                with client:
                    print('client connected!')
                    while True:
                        data = client.recv(1024)
                        if not data:
                            break
                        print('I received data!! :)')
                        print(str(data))

        except Exception as e:
            print(e)
            #self.signalException(e)

        finally:
            self.stop()

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