import subprocess
import json
import time
import math
import os
import sys
from threading import Thread

from PyQt5.QtCore import QObject, pyqtSignal

from src.utils.angles_3d_converter import Angles3DConverter


class OdasStream(QObject):

    signalOdasData = pyqtSignal(object)

    def __init__(self, odasPath, configPath, sleepTime, parent=None):
        super(OdasStream, self).__init__(parent)
        self.odasPath = odasPath
        self.configPath = configPath
        self.odasProcess = None
        self.sleepTime = sleepTime
        self.isRunning = False
        

    def start(self):
        Thread(target=self.run, args=()).start()


    def stop(self):
        if self.odasProcess and self.isRunning:
            print('Stopping Odas stream...')
            self.odasProcess.kill()
            print('Odas stream stopped.')
            self.isRunning = False

            if self.odasProcess.returncode and self.odasProcess.returncode != 0:
                raise Exception('ODAS exited with exit code {exitCode}'.format(exitCode=self.odasProcess.returncode))


    def run(self):
        self.__spawnSubProcess()

        stdout = []
        while True:

            if self.odasProcess.poll():
                self.stop()
                return

            line = self.odasProcess.stdout.readline().decode('UTF-8')

            if line:
                stdout.append(line)

            if len(stdout) > 8: # 8 because an object is 9 lines long.
                textoutput = '\n'.join(stdout)
                self.__parseOdasObject(textoutput)
                stdout.clear()

            time.sleep(self.sleepTime)


    # Spawn a sub process that execute odaslive.
    def __spawnSubProcess(self):
        if (not self.odasPath and not self.configPath):
            raise Exception('odasPath and configPath cannot be null or empty')

        print('ODAS stream starting...')
        self.odasProcess = subprocess.Popen([self.odasPath, '-c', self.configPath], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        self.isRunning = True
        

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