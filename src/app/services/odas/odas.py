import os
import json
import socket
from threading import Thread
from time import sleep

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.utils.json_utils import JsonUtils  
from src.utils.spherical_angles_converter import SphericalAnglesConverter
from src.app.services.odas.odasliveprocess.odas_live_process import OdasLiveProcess
from src.app.services.service.service_state import ServiceState


class Odas(QObject, Thread):
    '''
        Socket server that allow ODAS (https://github.com/introlab/odas) to send data back to our application.
        It accepts connections from ODAS and spawns workers for each connections that ODAS is trying to do.
    '''

    signalException = pyqtSignal(Exception)
    signalAudioData = pyqtSignal(bytes)
    signalPositionData = pyqtSignal(object)
    signalStateChanged = pyqtSignal(object)

    def __init__(self, hostIP, portPositions, portAudio, isVerbose=False, parent=None):
        QObject.__init__(self, parent)
        Thread.__init__(self)

        self.daemon = True
        
        self.host = hostIP
        self.portPositions = portPositions
        self.portAudio = portAudio
        self.isVerbose = isVerbose
        self.__workers = []

        self.isRunning = False
        self.state = ServiceState.STOPPED

        self.odasProcess = None
        self.odasPath =  ''
        self.micConfigPath = ''
        
        self.start()
    

    def stop(self):
        self.closeConnections()
        self.stopOdasLive()
        self.isRunning = False
        print('server stopped') if self.isVerbose else None


    def run(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socketPositions:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socketAudio:
                    socketPositions.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    socketPositions.bind((self.host, self.portPositions))
                    socketAudio.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    socketAudio.bind((self.host, self.portAudio))
                    socketPositions.listen()
                    socketAudio.listen()
                    self.isRunning = True
                    print('server is up!') if self.isVerbose else None

                    while True:
                        clientPositions, _ = socketPositions.accept()
                        print('client connected!') if self.isVerbose else None
                        clientAudio, _ = socketAudio.accept()
                        print('client connected!') if self.isVerbose else None
                        
                        if clientPositions and clientAudio:
                            
                            worker = self.__initWorker(clientPositions)
                            worker.start()
                            self.__workers.append(worker)

                            worker = self.__initWorker(clientAudio, 1024)
                            worker.start()
                            self.__workers.append(worker)

                            self.state = ServiceState.RUNNING
                            self.signalStateChanged.emit(ServiceState.RUNNING)

                        sleep(0.1)

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
        self.signalException.emit(e)
        
        if self.odasProcess:
            self.odasProcess.signalException.disconnect(self.odasLiveExceptionHandling)
            self.odasProcess = None
            self.closeConnections()

        self.state = ServiceState.STOPPED
        self.signalStateChanged.emit(ServiceState.STOPPED)


    def __initWorker(self, connection, bufferSize=None):
        worker = ClientHandler(connection, isVerbose=self.isVerbose, bufferSize=bufferSize)
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
                worker.join()
            
            self.__workers = []
                
        self.state = ServiceState.STOPPED
        self.signalStateChanged.emit(ServiceState.STOPPED)


    def startOdasLive(self, odasPath, micConfigPath):
        '''
            Spawn a sub process that execute odaslive.
        '''

        try:
        
            if not self.odasProcess:
                self.state = ServiceState.STARTING
                self.signalStateChanged.emit(ServiceState.STARTING)

                if not odasPath:
                    raise Exception('odasPath needs to be set in the settings')

                if not micConfigPath:
                    raise Exception('micConfigPath needs to be set in the settings')

                self.odasProcess = OdasLiveProcess(odasPath, micConfigPath)
                self.odasProcess.signalException.connect(self.odasLiveExceptionHandling)
                self.odasProcess.start()
                print('odas subprocess started...') if self.isVerbose else None

        except Exception as e:

            self.state = ServiceState.STOPPED
            self.signalStateChanged.emit(ServiceState.STOPPED)
            self.signalException.emit(e)


    def stopOdasLive(self):
        '''
            Stop the sub process spawned for ODAS.
        '''
        if self.odasProcess:
            if self.state == ServiceState.RUNNING:
                self.closeConnections()
            self.odasProcess.stop()
            self.odasProcess.signalException.disconnect(self.odasLiveExceptionHandling)
            self.odasProcess = None
            print('odas subprocess stopped...') if self.isVerbose else None


class ClientHandler(QObject, Thread):
    '''
        Workers that receives data from ODAS' (https://github.com/introlab/odas) sockets parse the data and return it back to this worker's server.
        Each worker handles a socket connection.
    '''

    signalConnectionClosed = pyqtSignal(object)
    signalAudio = pyqtSignal(bytes)
    signalPositions = pyqtSignal(object)

    def __init__(self, sock, bufferSize, isVerbose=False, parent=None):
        super(ClientHandler, self).__init__(parent)
        Thread.__init__(self)

        self.sock = sock
        self.isVerbose = isVerbose
        self.bufferSize = bufferSize
        self.isConnected = True
        self.daemon = True


    def stop(self):
        if self.sock:
            self.sock.close()
            self.isConnected = False
            self.sock = None
            self.join()
            print('connection closed') if self.isVerbose else None
            self.signalConnectionClosed.emit(self)

    def run(self):
        try:
            while True:
                if not self.isConnected or not self.sock:
                    self.isConnected = False
                    break
                if self.bufferSize:
                    data = self.sock.recv(self.bufferSize, socket.MSG_WAITALL)
                else:
                    data = self.sock.recv(10000)
                # If there is no data incomming close the stream.
                if not data:
                    self.isConnected = False
                    break

                if JsonUtils.isJson(data):
                    self.__parseOdasObject(data)
                elif len(data) == self.bufferSize:
                    self.signalAudio.emit(data)
                            
                sleep(0.00001)

        except Exception as e:
            self.signalConnectionClosed.emit(self)

    
    # Parse every Odas event.
    def __parseOdasObject(self, jsonBytes):
        parsedJson = json.loads(str(jsonBytes, 'utf-8'))
        jsonSources = parsedJson['src']

        sources = []
        for index, jsonSource in enumerate(jsonSources):
            jsonSource['azimuth'] = SphericalAnglesConverter.getAzimuthFromPosition(jsonSource['x'], jsonSource['y'])
            jsonSource['elevation'] = SphericalAnglesConverter.getElevationFromPosition(jsonSource['x'], jsonSource['y'], jsonSource['z'])
            sources.append(jsonSource)

        if sources:
            self.signalPositions.emit(sources)
