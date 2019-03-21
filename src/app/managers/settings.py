import os
from pathlib import Path

from PyQt5.QtCore import QSettings


class Settings:
    
    rootDirectory = str(Path(__file__).resolve().parents[2])

    qSettings = QSettings('Groupe RendezVous', 'App')

    def __init__(self):
        defaultOutputFolder = self.getValue('defaultOutputFolder')
        if not defaultOutputFolder:
            self.setValue('defaultOutputFolder', )


    @staticmethod
    def getValue(key):
        return Settings.qSettings.value(key)


    @staticmethod
    def setValue(key, value):
        Settings.qSettings.setValue(key, value)
