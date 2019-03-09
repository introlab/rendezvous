from PyQt5.QtCore import QSettings


class SettingsManager:

    qSettings = QSettings("Groupe RendezVous", "App")

    def __init__(self):
        pass


    @staticmethod
    def getValue(key):
        return SettingsManager.qSettings.value(key)


    @staticmethod
    def setValue(key, value):
        SettingsManager.qSettings.setValue(key, value)
