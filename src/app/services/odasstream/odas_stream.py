import subprocess
import json
import time
import re
from threading import Thread

from PyQt5.QtCore import QObject, pyqtSignal

from src.utils.spherical_angles_converter import SphericalAnglesConverter
from src.utils.file_helper import FileHelper


class OdasStream(QObject):

    signalOdasData = pyqtSignal(object)
    signalOdasException = pyqtSignal(Exception)

    def __init__(self, parent=None):
        super(OdasStream, self).__init__(parent)
        self.odasProcess = None
        self.isRunning = False
        

    def start(self, odasPath, micConfigPath):
        print('Starting Odas stream...')

        try:

            if not odasPath:
                raise Exception('odasPath needs to be set in the settings')

            if not micConfigPath:
                raise Exception('micConfigPath needs to be set in the settings')

            # Read config file to get sample rate for while True sleepTime
            line = FileHelper.getLineFromFile(micConfigPath, 'fS')
            if not line:
                raise Exception('sample rate not found in ', micConfigPath)

            # Extract the sample rate from the string and convert to an Integer
            sampleRate = int(re.sub('[^0-9]', '', line.split('=')[1]))
            sleepTime = 1 / sampleRate

            Thread(target=self.run, args=(odasPath, micConfigPath, sleepTime)).start()

        except Exception as e:
            
            self.isRunning = False
            self.signalOdasException.emit(e)


    def stop(self):
        self.isRunning = False


    def run(self, odasPath, micConfigPath, sleepTime):
        self.__spawnSubProcess(odasPath, micConfigPath)

        print('ODAS stream started')

        self.isRunning = True

        stdout = []
        while self.isRunning:

            if self.odasProcess.poll():
                self.stop()
                break

            line = self.odasProcess.stdout.readline().decode('UTF-8')

            if line:
                stdout.append(line)

            if len(stdout) > 8: # 8 because an object is 9 lines long.
                textoutput = '\n'.join(stdout)
                self.__parseOdasObject(textoutput)
                stdout.clear()

            time.sleep(sleepTime)
   
        self.odasProcess.kill()
        if self.odasProcess.returncode and self.odasProcess.returncode != 0:
            self.isRunning = False
            e = Exception('ODAS exited with exit code {exitCode}'.format(exitCode=self.odasProcess.returncode))
            self.signalOdasException.emit(e)

        print('ODAS process terminated')


    # Spawn a sub process that execute odaslive.
    def __spawnSubProcess(self, odasPath, micConfigPath):
        print('ODAS stream starting...')
        self.odasProcess = subprocess.Popen([odasPath, '-c', micConfigPath], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        

    # Parse every Odas event 
    def __parseOdasObject(self, jsonText):
        parsedJson = json.loads(jsonText)
        jsonSources = parsedJson['src']

        sources = {}
        for index, jsonSource in enumerate(jsonSources):
            jsonSource['azimuth'] = SphericalAnglesConverter.getAzimuthFromPosition(jsonSource['x'], jsonSource['y'])
            jsonSource['elevation'] = SphericalAnglesConverter.getElevationFromPosition(jsonSource['x'], jsonSource['y'], jsonSource['z'])
            sources[index] = jsonSource

        if sources:
            self.signalOdasData.emit(sources)