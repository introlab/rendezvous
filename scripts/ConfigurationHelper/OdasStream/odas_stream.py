import subprocess as sp
import json
import math
from threading import Thread
from time import sleep

import numpy


class OdasStream:

    def __init__(self, odasPath, configPath):
        self.odasPath = odasPath
        self.configPath = configPath


    # Create and start thread for capturing odas stream
    def start(self):
        self.__spawnOdasProcess()
        self.__processOdasOutput()


    # Stop odas Thread
    def stop(self):
        print('Stopping odas stream...')
        self.subProcess.terminate()
        print('odas stream stopped.')


    def __spawnOdasProcess(self):
        if (not self.odasPath and not self.configPath):
            raise Exception('odasPath and configPath cannot be null or empty')

        print('ODAS stream starting...')
        self.subProcess = sp.Popen([self.odasPath, '-c', self.configPath], shell=False, stdout=sp.PIPE)
        sp.call([self.odasPath, '-c', self.configPath])


    def __processOdasOutput(self):

        stdout = []
        stdoutobj = []

        # Need to check if device detected
        while True:
            line = self.subProcess.stdout.readline().decode('UTF-8')
            stdoutobj.append(line)
            if len(stdoutobj) > 8:
                stdout.extend(stdoutobj)
                stdoutobj.clear()
            if stdout:
                textoutput = '\n'.join(stdout)
                self.__parseJsonStream(textoutput)
                stdout.clear()
            
            sleep(0.01) # sleep for 1ms


    def __parseJsonStream(self, jsonText):

        print(jsonText)
        self.data = []
        self.sources = {'1':[], '2':[], '3':[], '4':[]}

        parsed_json = json.loads(jsonText)
        timeStamp = parsed_json['timeStamp']
        src = parsed_json['src']

        self.data.append([timeStamp, src])
