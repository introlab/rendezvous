import os
import json
import socket
from threading import Thread
from time import sleep
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.utils.json_utils import JsonUtils  
from src.utils.spherical_angles_converter import SphericalAnglesConverter
from src.app.services.odas.odasliveprocess.odas_live_process import OdasLiveProcess

class Odas(QObject, Thread):

    signalException = pyqtSignal(Exception)
    signalAudioData = pyqtSignal(bytes)
    signalPositionData = pyqtSignal(object)
    signalClientsConnected = pyqtSignal(bool)

    def __init__(self, hostIP, port, isVerbose=False, parent=None):
        super(Odas, self).__init__(parent)
        Thread.__init__(self)

        self.daemon = True
        
        self.host = hostIP
        self.port = port
        self.isVerbose = isVerbose
        self.__workers = []

        self.isRunning = False
        self.isConnected = False

        self.odasProcess = None
        self.odasPath =  ''
        self.micConfigPath = ''
    

    def stop(self):
        self.closeConnections()
        self.stopOdasLive()
        self.isRunning = False
        print('server stopped') if self.isVerbose else None


    def run(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((self.host, self.port))
                # Wait for 2 clients max.
                sock.listen(2)
                self.isRunning = True
                print('server is up!') if self.isVerbose else None

                while True:
                    clientConnection, _ = sock.accept()
                    if clientConnection:
                        self.isConnected = True
                        print('client connected!') if self.isVerbose else None
                        worker = self.__initWorker(clientConnection)
                        worker.start()
                        self.__workers.append(worker)
                        self.signalClientsConnected.emit(True)

                    sleep(0.00001)

        except Exception as e:
            self.closeConnections()
            self.stopOdasLive()
            self.signalException.emit(e)

        finally:
            self.isRunning = False


    @pyqtSlot(object)
    def positionsReceived(self, data):
        if data:
            self.signalPositionData.emit(data)


    @pyqtSlot(bytes)
    def audioReceived(self, data):
        if data:
            self.signalAudioData.emit(data)


    @pyqtSlot(object)
    def workerTerminated(self, worker):
        if worker in self.__workers:
            self.__workers.remove(worker)


    @pyqtSlot(Exception)
    def odasLiveExceptionHandling(self, e):
        if self.odasProcess:
            self.odasProcess.signalException.disconnect(self.odasLiveExceptionHandling)
            self.odasProcess = None
            self.closeConnections()
        self.signalException.emit(e)


    def __initWorker(self, connection):
        worker = ClientHandler(connection, isVerbose=self.isVerbose)
        worker.signalAudio.connect(self.audioReceived)
        worker.signalPositions.connect(self.positionsReceived)
        worker.signalConnectionClosed.connect(self.workerTerminated)
        return worker


    def closeConnections(self):
        if self.__workers:
            for worker in self.__workers:
                worker.signalAudio.disconnect(self.audioReceived)
                worker.signalPositions.disconnect(self.positionsReceived)
                worker.signalConnectionClosed.disconnect(self.workerTerminated)
                worker.stop()
            
            self.__workers = []
                

        self.signalClientsConnected.emit(False)


    # Spawn a sub process that execute odaslive.
    def startOdasLive(self, odasPath, micConfigPath):
        if not self.odasProcess:
            if not odasPath:
                raise Exception('odasPath needs to be set in the settings')

            if not micConfigPath:
                raise Exception('micConfigPath needs to be set in the settings')

            self.odasProcess = OdasLiveProcess(odasPath, micConfigPath)
            self.odasProcess.signalException.connect(self.odasLiveExceptionHandling)
            self.odasProcess.start()
            print('odas subprocess started...') if self.isVerbose else None


    # Stop the sub process.
    def stopOdasLive(self):
        if self.odasProcess:
            if self.isConnected:
                self.closeConnections()
            self.odasProcess.signalException.disconnect(self.odasLiveExceptionHandling)
            self.odasProcess.stop()
            self.odasProcess = None
            print('odas subprocess stopped...') if self.isVerbose else None


class ClientHandler(QObject, Thread):

    signalConnectionClosed = pyqtSignal(object)
    signalAudio = pyqtSignal(bytes)
    signalPositions = pyqtSignal(object)

    def __init__(self, sock, isVerbose=False, parent=None):
        super(ClientHandler, self).__init__(parent)
        Thread.__init__(self)

        self.sock = sock
        self.isVerbose = isVerbose
        self.isConnected = True
        self.daemon = True


    def stop(self):
        if self.sock:
            self.sock.close()
            self.isConnected = False
            self.sock = None

            print('connection closed') if self.isVerbose else None
            self.signalConnectionClosed.emit(self)

    def run(self):
        try:
            while True:
                if not self.isConnected or not self.sock:
                    self.isConnected = False
                    return
                # 1024 because this is the minimum Odas send through the socket.
                data = self.sock.recv(1024)
                # If there is no data incomming close the stream.
                if not data:
                    self.isConnected = False
                    return
                            
                if JsonUtils.isJson(data):
                    self.__parseOdasObject(data)
                else:
                    # Print(data).
                    self.signalAudio.emit(data)
                            
                sleep(0.00001)

        except Exception as e:
            self.signalConnectionClosed.emit(self)

    
    # Parse every Odas event.
    def __parseOdasObject(self, jsonBytes):
        parsedJson = json.loads(str(jsonBytes, 'utf-8'))
        jsonSources = parsedJson['src']

        sources = {}
        for index, jsonSource in enumerate(jsonSources):
            jsonSource['azimuth'] = np.rad2deg(SphericalAnglesConverter.getAzimuthFromPosition(jsonSource['x'], jsonSource['y']))
            jsonSource['elevation'] = np.rad2deg(SphericalAnglesConverter.getElevationFromPosition(jsonSource['x'], jsonSource['y'], jsonSource['z']))
            sources[index] = jsonSource

        if sources:
            self.signalPositions.emit(sources)

