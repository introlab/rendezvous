import sys

from PyQt5.QtWidgets import QApplication

import context
from src.app.main_window import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.showMaximized() 
    exit(app.exec_())
