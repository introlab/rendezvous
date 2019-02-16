import subprocess as sp
import json
from time import sleep
import os
import sys

from FileHandler.file_handler import FileHandler

workingDirectory = os.path.dirname(os.path.realpath(sys.argv[0]))

class OdasStream:

    def __init__(self, odasPath, configPath, options={}):
        self.odasPath = odasPath
        self.configPath = configPath
        self.sleepTime = options['sleepTime']
        self.maxChunkSize = options['chunkSize']
        self.isRunning = False
        self.data = []
        self.currentChunkSize = 0


    # public function to start the stream
    def start(self):
        self.__spawnSubProcess()
        self.__processOutput()


    # public function to stop the stream
    def stop(self):
        print('Stopping odas stream...')
        self.subProcess.kill()
        print('odas stream stopped.')
        self.isRunning = False

        if self.subProcess and self.subProcess.returncode and self.subProcess.returncode != 0:
            raise Exception('ODAS exited with exit code {exitCode}'.format(exitCode=self.subProcess.returncode))


    # spawn a sub process that execute odaslive.
    def __spawnSubProcess(self):
        if (not self.odasPath and not self.configPath):
            raise Exception('odasPath and configPath cannot be null or empty')

        print('ODAS stream starting...')
        self.subProcess = sp.Popen([self.odasPath, '-c', self.configPath], shell=False, stdout=sp.PIPE, stderr=sp.PIPE)
        sp.call([self.odasPath, '-c', self.configPath])

        self.isRunning = True


    #  get odas' events and store it in self.data
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
                
                # backup of events for each chunk of events defined in self.chunkSize
                if (self.currentChunkSize > self.maxChunkSize):
                    self.__backupEvents()

                stdout.clear()
            
            sleep(self.sleepTime)


    # backup of events in ODASOutput.json
    def __backupEvents(self):
        if self.data:
            fileName = 'ODASOutput.json'
            streamOutputPath = os.path.join(workingDirectory, fileName)
            FileHandler.writeJsonToFile(streamOutputPath, self.data)
            self.currentChunkSize = 0


    # parse every event in object and store it.
    def __parseJsonStream(self, jsonText):
        print(jsonText)
        parsed_json = json.loads(jsonText)
        self.data.append([parsed_json])
        self.currentChunkSize += 1
