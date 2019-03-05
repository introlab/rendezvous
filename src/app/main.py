import sys
import re
import os

from PyQt5.QtWidgets import QApplication

import context
from src.app.main_modules.main_window import MainWindow
from src.app.argsparser.args_parser import ArgsParser 
from src.app.odasstream.odas_stream import OdasStream
from src.utils.file_helper import FileHelper

if __name__ == '__main__':
    parser = ArgsParser()
    args = parser.args

    # Read config file to get sample rate for while True sleepTime
    line = FileHelper.getLineFromFile(args.configPath, 'fS')

    # Extract the sample rate from the string and convert to an Integer
    sampleRate = int(re.sub('[^0-9]', '', line.split('=')[1]))
    sleepTime = 1 / sampleRate

    stream = OdasStream(args.odasPath, args.configPath, sleepTime)

    app = QApplication(sys.argv)
    main_window = MainWindow(stream)
    main_window.show()
    exit(app.exec_())