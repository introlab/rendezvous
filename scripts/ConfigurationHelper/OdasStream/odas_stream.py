import subprocess as sp
import json
import math
from threading import Thread
from time import sleep

import numpy


class OdasStream:

    def __init__(self, odasPath, configPath, sleepTime):
        self.odasPath = odasPath
        self.configPath = configPath
        self.sleepTime = sleepTime
        self.isRunning = False


    # Create and start thread for capturing odas stream
    def start(self):
        self.__spawnOdasProcess()
        self.__processOdasOutput()


    # Stop odas Thread
    def stop(self):
        print('Stopping odas stream...')
        self.subProcess.terminate()
        print('odas stream stopped.')
        self.isRunning = False

        if self.subProcess.returncode != 0:
            raise Exception('ODAS exited with exit code {exitCode}'.format(exitCode=self.subProcess.returncode))


    def __spawnOdasProcess(self):
        if (not self.odasPath and not self.configPath):
            raise Exception('odasPath and configPath cannot be null or empty')

        print('ODAS stream starting...')
        self.subProcess = sp.Popen([self.odasPath, '-c', self.configPath], shell=False, stdout=sp.PIPE, stderr=sp.PIPE)
        sp.call([self.odasPath, '-c', self.configPath])

        self.isRunning = True


    def __processOdasOutput(self):

        stdout = []
        stdoutobj = []

        # Need to check if device detected
        while True:
            if self.subProcess.poll():
                self.stop()

            line = self.subProcess.stdout.readline().decode('UTF-8')
            if line:
                stdoutobj.append(line)
            if len(stdoutobj) > 8:
                stdout.extend(stdoutobj)
                stdoutobj.clear()
            if stdout:
                textoutput = '\n'.join(stdout)
                self.__parseJsonStream(textoutput)
                stdout.clear()

            
            sleep(self.sleepTime)


    def __parseJsonStream(self, jsonText):

        print(jsonText)
        self.data = []
        self.sources = {'1':[], '2':[], '3':[], '4':[]}

        parsed_json = json.loads(jsonText)
        timeStamp = parsed_json['timeStamp']
        src = parsed_json['src']

        self.data.append([timeStamp, src])
