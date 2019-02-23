from PyQt5 import QtWidgets
from mainwindow import Ui_MainWindow
import sys
 
class ExampleApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(ExampleApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

def main():
    app = QtWidgets.QApplication([])
    application = ExampleApp()
    application.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
 