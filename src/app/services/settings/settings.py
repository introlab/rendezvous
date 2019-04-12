import os
from pathlib import Path

from PyQt5.QtCore import QSettings

from src.app.services.speechtotext.speech_to_text import EncodingTypes, LanguageCodes, Models


class Settings(object):
    
    rootDirectory = str(Path(__file__).resolve().parents[4])

    qSettings = QSettings('Groupe RendezVous', 'App')

    def __init__(self):
        defaultOutputFolder = self.getValue('defaultOutputFolder')
        if not defaultOutputFolder:
            self.setValue('defaultOutputFolder', self.rootDirectory)

        speechToTextSampleRate = self.getValue('speechToTextSampleRate')
        if not speechToTextSampleRate:
            self.setValue('speechToTextSampleRate', 48000)

        speechToTextEncoding = self.getValue('speechToTextEncoding')
        if not speechToTextEncoding:
            self.setValue('speechToTextEncoding', EncodingTypes.ENCODING_UNSPECIFIED)

        speechToTextLanguage = self.getValue('speechToTextLanguage')
        if not speechToTextLanguage:
            self.setValue('speechToTextLanguage', LanguageCodes.FR_CA)

        speechToTextModel = self.getValue('speechToTextModel')
        if not speechToTextModel:
            self.setValue('speechToTextModel', Models.DEFAULT)

        speechToTextEnhanced = self.getValue('speechToTextEnhanced')
        if not speechToTextEnhanced:
            self.setValue('speechToTextEnhanced', False)

        automaticTranscription = self.getValue('automaticTranscription')
        if not automaticTranscription:
            self.setValue('automaticTranscription', False)


    @staticmethod
    def getValue(key):
        return Settings.qSettings.value(key)


    @staticmethod
    def setValue(key, value):
        Settings.qSettings.setValue(key, value)
    