import os
from threading import Thread
import subprocess
from time import sleep

from PyQt5.QtCore import QObject, pyqtSignal

class OdasLiveProcess(QObject, Thread):

    signalException = pyqtSignal(Exception)

    def __init__(self, odasPath, micConfigPath, parent=None):
        super(OdasLiveProcess, self).__init__(parent)
        Thread.__init__(self)

        self.odasPath = odasPath
        self.micConfigPath = micConfigPath
        self.process = None


    def stop(self):
        if self.process:
            self.process.terminate()
        self.join()


    def run(self):
        try:
            if not os.path.exists(self.odasPath):
                raise Exception('no file found at {}'.format(self.odasPath))

            if not os.path.exists(self.micConfigPath):
                raise Exception('no file found at {}'.format(self.micConfigPath))

            self.process = subprocess.Popen([self.odasPath, '-c', self.micConfigPath], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            while True:
                if self.process and self.process.poll():
                    self.process.terminate()
                    if self.process.returncode != 0:
                        raise Exception('ODAS exited with exit code {}'.format(self.process.returncode))

                sleep(1)

        except Exception as e:
            self.signalException.emit(e)
