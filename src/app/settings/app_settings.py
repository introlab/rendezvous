from PyQt5.QtCore import QSettings


class AppSettings:

    qSettings = QSettings("Groupe RendezVous", "App")

    def __init__(self):
        pass


    @staticmethod
    def getValue(key):
        return AppSettings.qSettings.value(key)


    @staticmethod
    def setValue(key, value):
        AppSettings.qSettings.setValue(key, value)
