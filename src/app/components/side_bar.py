from PyQt5.QtWidgets import QListWidget, QListWidgetItem 
from PyQt5.QtCore import QSize


class SideBar(QListWidget):

    def __init__(self, parent=None):
        super(SideBar, self).__init__(parent)

        self.setFixedWidth(150)
        self.setStyleSheet('''
            QListWidget {
                background-color: #4a4a4a; 
                color: #ffffff;
                selection-background-color: #3a3a3a;
                selection-color: #016735;
            }

            QListWidget::item::hover {
                background-color: #3a3a3a;
                color: #016735;
            }
        ''') 
        

    # Override of insertItem from QListWidget to customize items.
    def addItem(self, name):
        item = QListWidgetItem(name)
        item.setSizeHint(QSize(0, 40))
        super().addItem(item)

