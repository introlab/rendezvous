import os
from pathlib import Path

from PyQt5.QtCore import QSettings

from src.app.services.speechtotext.speech_to_text import EncodingTypes, LanguageCodes, Models


class Settings(object):
    
    rootDirectory = str(Path(__file__).resolve().parents[4])

    qSettings = QSettings('Groupe RendezVous', 'App')

    def __init__(self):
        defaultOutputFolder = self.getValue('defaultOutputFolder')
        if defaultOutputFolder == None:
            self.setValue('defaultOutputFolder', self.rootDirectory)

        speechToTextSampleRate = self.getValue('speechToTextSampleRate')
        if speechToTextSampleRate == None:
            self.setValue('speechToTextSampleRate', 48000)

        speechToTextEncoding = self.getValue('speechToTextEncoding')
        if speechToTextEncoding == None:
            self.setValue('speechToTextEncoding', EncodingTypes.ENCODING_UNSPECIFIED)

        speechToTextLanguage = self.getValue('speechToTextLanguage')
        if speechToTextLanguage == None:
            self.setValue('speechToTextLanguage', LanguageCodes.FR_CA)

        speechToTextModel = self.getValue('speechToTextModel')
        if speechToTextModel == None:
            self.setValue('speechToTextModel', Models.DEFAULT)

        speechToTextEnhanced = self.getValue('speechToTextEnhanced')
        if speechToTextEnhanced == None:
            self.setValue('speechToTextEnhanced', 0)

        automaticTranscription = self.getValue('automaticTranscription')
        if automaticTranscription == None:
            self.setValue('automaticTranscription', 0)


    @staticmethod
    def getValue(key):
        return Settings.qSettings.value(key)


    @staticmethod
    def setValue(key, value):
        Settings.qSettings.setValue(key, value)
    