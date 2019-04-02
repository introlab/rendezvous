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
        self.stopCommand = False


    def stop(self):
        if self.process:
            self.process.terminate()
        self.stopCommand = True
        self.join()

    def run(self):
        try:
            if not os.path.exists(self.odasPath):
                raise Exception('no file found at {}'.format(self.odasPath))

            if not os.path.exists(self.micConfigPath):
                raise Exception('no file found at {}'.format(self.micConfigPath))

            self.process = subprocess.Popen([self.odasPath, '-c', self.micConfigPath], shell=False)

            while True:
                if self.process and self.process.poll():
                    if self.process.returncode == 1 or self.process.returncode == -1:
                        raise Exception('ODAS exited with exit code {}'.format(self.process.returncode))
                    break
                
                elif not self.process or self.stopCommand:
                    break

                sleep(1)

        except Exception as e:
            self.signalException.emit(e)
