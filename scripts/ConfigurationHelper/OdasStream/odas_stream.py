import subprocess as sp
import json
import math
from time import sleep
import os
import sys

import numpy

from FileHandler.file_handler import FileHandler

workingDirectory = os.path.dirname(os.path.realpath(sys.argv[0]))

class OdasStream:

    def __init__(self, odasPath, configPath, sleepTime):
        self.odasPath = odasPath
        self.configPath = configPath
        self.sleepTime = sleepTime
        self.isRunning = False
        self.data = []
        self.backupCounter = 0


    # Create and start thread for capturing odas stream
    def start(self):
        self.__spawnSubProcess()
        self.__processOutput()


    # Stop odas Thread
    def stop(self):
        print('Stopping odas stream...')
        self.subProcess.kill()
        print('odas stream stopped.')
        self.isRunning = False

        if self.subProcess and self.subProcess.returncode and self.subProcess.returncode != 0:
            raise Exception('ODAS exited with exit code {exitCode}'.format(exitCode=self.subProcess.returncode))


    def __spawnSubProcess(self):
        if (not self.odasPath and not self.configPath):
            raise Exception('odasPath and configPath cannot be null or empty')

        print('ODAS stream starting...')
        self.subProcess = sp.Popen([self.odasPath, '-c', self.configPath], shell=False, stdout=sp.PIPE, stderr=sp.PIPE)
        sp.call([self.odasPath, '-c', self.configPath])

        self.isRunning = True


    def __processOutput(self):

        stdout = []

        # Need to check if device detected
        while True:
            if self.subProcess.poll():
                self.stop()
                return

            line = self.subProcess.stdout.readline().decode('UTF-8')
            if line:
                stdout.append(line)
            if len(stdout) > 8: # 8 because an object is 9 lines long.
                textoutput = '\n'.join(stdout)
                self.__parseJsonStream(textoutput)
                stdout.clear()

            
            sleep(self.sleepTime)


    def __parseJsonStream(self, jsonText):

        print(jsonText)
        parsed_json = json.loads(jsonText)
        self.data.append([parsed_json])
        self.backupCounter += 1

        # backup of events every 500 events.
        if (self.backupCounter > 500):
            if self.data:
                fileName = 'ODASOutput.json'
                streamOutputPath = os.path.join(workingDirectory, fileName)
                FileHandler.writeJsonToFile(streamOutputPath, self.data)
                self.backupCounter = 0
