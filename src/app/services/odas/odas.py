
import os
import json
import re
import subprocess
import socket
from threading import Thread
from time import sleep

from PyQt5.QtCore import QObject, pyqtSignal

from src.utils.angles_3d_converter import Angles3DConverter
from src.utils.file_helper import FileHelper

# Read config file to get sample rate for while True sleepTime
        # line = FileHelper.getLineFromFile(micConfigPath, 'fS')
        # if not line:
        #     raise Exception('sample rate not found in ', micConfigPath)

        # # Extract the sample rate from the string and convert to an Integer
        # sampleRate = int(re.sub('[^0-9]', '', line.split('=')[1]))
        # sleepTime = 1 / sampleRate

    # def run(self, odasPath, micConfigPath, sleepTime):
    #     try:
    #         self.__spawnSubProcess(odasPath, micConfigPath)
    #         stdout = []
    #         while self.isRunning:

    #             if self.odasProcess.poll():
    #                 self.stop()
    #                 break

    #             line = self.odasProcess.stdout.readline().decode('UTF-8')
    #             # at this point odaslive is ready to serve
    #             self.isRunning = True

    #             if line:
    #                 stdout.append(line)

    #             if len(stdout) > 8: # 8 because an object is 9 lines long.
    #                 textoutput = '\n'.join(stdout)
    #                 self.__parseOdasObject(textoutput)
    #                 stdout.clear()

    #             sleep(sleepTime)
    
    #         self.odasProcess.kill()
    #         if self.odasProcess.returncode and self.odasProcess.returncode != 0:
    #             raise Exception('ODAS exited with exit code {exitCode}'.format(exitCode=self.odasProcess.returncode))
        
    #     except Exception as e:
    #         self.signalException.emit(e)

    #     finally:
    #         self.isRunning = False
class Odas(QObject, Thread):

    signalException = pyqtSignal(Exception)
    signalAudioData = pyqtSignal(bytes)
    signalPositionData = pyqtSignal(object)
    signalData = pyqtSignal(object)
    signalClientConnected = pyqtSignal(bool)

    def __init__(self, hostIP, port, isVerbose=False, parent=None):
        super(Odas, self).__init__(parent)
        Thread.__init__(self)

        self.daemon = True
        
        self.host = hostIP
        self.port = port
        self.isVerbose = isVerbose

        self.isRunning = False
        self.isConnected = False

        self.clientConnection = None
        self.odasProcess = None
        self.odasPath =  ''
        self.micConfigPath = ''
    

    def stop(self):
        self.closeConnection()
        self.stopOdasLive()
        self.isRunning = False
        self.signalClientConnected.emit(False)
        print('server stopped') if self.isVerbose else None


    def run(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((self.host, self.port))
                sock.listen()
                self.isRunning = True
                print('server is up!') if self.isVerbose else None

                while True:
                    self.clientConnection, _ = sock.accept()
                    self.signalClientConnected.emit(True)
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
                            self.signalData.emit(data)
                            sleep(0.00001)
                    
                    sleep(0.00001)

        except Exception as e:
            self.closeConnection()
            self.stopOdasLive()
            self.signalClientConnected.emit(False)

            self.signalException.emit(e)

        finally:
            self.isRunning = False


    def closeConnection(self):
        if self.clientConnection:
            self.clientConnection.close()
            self.isConnected = False
            self.clientConnection = None
            print('connection closed') if self.isVerbose else None


    # Spawn a sub process that execute odaslive.
    def startOdasLive(self, odasPath, micConfigPath):
        if not self.odasProcess:
            if not odasPath:
                raise Exception('odasPath needs to be set in the settings')

            if not micConfigPath:
                raise Exception('micConfigPath needs to be set in the settings')

            self.odasProcess = subprocess.Popen([odasPath, '-c', micConfigPath], shell=False)
            print('odas subprocess started...') if self.isVerbose else None


    # stop the sub process
    def stopOdasLive(self):
        if self.odasProcess:
            self.odasProcess.kill()
            self.odasProcess = None
            print('odas subprocess stopped...') if self.isVerbose else None


    # Parse every Odas event 
    def __parseOdasObject(self, jsonText):
        parsedJson = json.loads(jsonText)
        jsonSources = parsedJson['src']

        sources = {}
        for index, jsonSource in enumerate(jsonSources):
            jsonSource['azimuth'] = Angles3DConverter.azimuthCalculation(jsonSource['x'], jsonSource['y'])
            jsonSource['elevation'] = Angles3DConverter.elevationCalculation(jsonSource['x'], jsonSource['y'], jsonSource['z'])
            sources[index] = jsonSource

        if sources:
            self.signalOdasData.emit(sources)

