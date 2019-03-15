from PyQt5.QtCore import QSettings


class Settings:

    qSettings = QSettings('Groupe RendezVous', 'App')

    def __init__(self):
        pass


    @staticmethod
    def getValue(key):
        return Settings.qSettings.value(key)


    @staticmethod
    def setValue(key, value):
        Settings.qSettings.setValue(key, value)
