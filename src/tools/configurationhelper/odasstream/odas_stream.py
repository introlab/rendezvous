import subprocess as sp
import json
from time import sleep
import os
import sys
import signal

from utils.file_helper import FileHelper

workingDirectory = os.path.dirname(os.path.realpath(sys.argv[0]))


class AlarmException(Exception):
    pass


class OdasStream:

    def __init__(self, odasPath, configPath, sleepTime, options={'chunkSize' : 500}):
        self.odasPath = odasPath
        self.configPath = configPath
        self.sleepTime = sleepTime
        self.maxChunkSize = options['chunkSize']
        self.isRunning = False
        self.data = []
        self.currentChunkSize = 0


    # public function to start the stream
    def start(self, executionTime=-1):
        if (executionTime != -1):
            signal.signal(signal.SIGALRM, self.__alarmCallback)
            # signal.alarm function takes time in seconds.
            signal.alarm(executionTime * 60)

        self.__spawnSubProcess()
        self.__processOutput()


    # public function to stop the stream
    def stop(self):
        if self.subProcess:
            print('Stopping odas stream...')
            self.subProcess.kill()
            print('odas stream stopped.')
            self.isRunning = False

            if self.subProcess.returncode and self.subProcess.returncode != 0:
                raise Exception('ODAS exited with exit code {exitCode}'.format(exitCode=self.subProcess.returncode))


    def __alarmCallback(self, signum, frame):
        # to be sure events are saved in a file.
        self.__backupEvents()
        self.stop()
        raise AlarmException('Execution Timeout')


    # spawn a sub process that execute odaslive.
    def __spawnSubProcess(self):
        if (not self.odasPath and not self.configPath):
            raise Exception('odasPath and configPath cannot be null or empty')

        print('ODAS stream starting...')
        self.subProcess = sp.Popen([self.odasPath, '-c', self.configPath], shell=False, stdout=sp.PIPE, stderr=sp.PIPE)

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
            FileHelper.writeJsonFile(streamOutputPath, self.data)
            self.currentChunkSize = 0


    # parse every event in object and store it.
    def __parseJsonStream(self, jsonText):
        parsed_json = json.loads(jsonText)

        src = parsed_json['src']
        timestamp = parsed_json['timeStamp']
        activeSources = {'timestamp' : timestamp, 'src' : {}}
        for index, source in enumerate(src):
            # if id equals zero that means the source is innactive.
            if source['id'] != 0:
                activeSources['src'][index + 1] = source

        # if there is an active source in the event.
        if activeSources['src'] != {}:
            self.data.append([activeSources])
            self.currentChunkSize += 1
