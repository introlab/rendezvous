import io
import os
from enum import Enum, unique, auto

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from google.cloud import speech


@unique
class EncodingTypes(Enum):
    ENCODING_UNSPECIFIED = 'ENCODING_UNSPECIFIED'
    FLAC = 'FLAC'
    AMR = 'AMR'
    AMR_WB = 'AMR_WB'
    LINEAR16 = 'LINEAR16'
    OGG_OPUS = 'OGG_OPUS'
    SPEEX_WITH_HEADER_BYTE = 'SPEEX_WITH_HEADER_BYTE'


@unique
class LanguageCodes(Enum):
    FR_CA = 'fr-CA'
    EN_CA = 'en-CA'


@unique
class Models(Enum):
    DEFAULT = 'default'
    COMMAND_AND_SEARCH = 'command_and_search'
    PHONE_CALL = 'phone_call'
    VIDEO = 'video' 


class SpeechToText(QObject):

    transcriptionReady = pyqtSignal(str)
    exception = pyqtSignal(Exception)

    # All the parameters needed to perform a transcription.
    __config = {'audioDataPath' : '',
                'encoding' : None,
                'enhanced' : False,
                'languageCode' : None,
                'model' : None,
                'outputFolder' : '',
                'sampleRate' : 0,
                'serviceAccountPath' : ''}

    # Valid range accepted by the Google API.
    __minSampleRate = 8000
    __maxSampleRate = 48000
    # Value we are most likely to use.
    __defaultSampleRate = 48000       

    def setConfig(self, config):
        self.__config = config


    def getMinSampleRate(self):
        return self.__minSampleRate


    def getMaxSampleRate(self):
        return self.__maxSampleRate


    def getDefaultSampleRate(self):
        return self.__defaultSampleRate


    def resquestTranscription(self):
        try:
            # Validations before starting transcription procedure.
            if not os.path.exists(self.__config['serviceAccountPath']):
                raise Exception('No Google Service Account File found at : {}'.format(self.__config['serviceAccountPath']))

            if not os.path.exists(self.__config['audioDataPath']):
                raise Exception('No Audio Data found at : {}'.format(self.__config['audioDataPath']))

            if not os.path.exists(self.__config['outputFolder']):
                raise Exception('No default output folder  found at : {}'.format(self.__config['outputFolder']))

            if not self.__config['encoding'] in [encodingType.value for encodingType in EncodingTypes]:
                raise Exception('{} is not a supported encoding format'.format(self.__config['encoding']))
            
            # Range accepted by the API.
            if not self.__config['sampleRate'] in range(8000, 48001):
                raise Exception('Sample rate value {} is not in valid range'.format(self.__config['sampleRate']))

            if not self.__config['languageCode'] in [languageCode.value for languageCode in LanguageCodes]:
                raise Exception('{} is not a supported language code'.format(self.__config['languageCode']))
            
            if not self.__config['model'] in [model.value for model in Models]:
                raise Exception('{} is not a supported model'.format(self.__config['model']))

            # Instantiates a client.
            client = speech.SpeechClient.from_service_account_json(self.__config['serviceAccountPath'])

            # The name of the audio data to transcribe.
            fileName = os.path.join(self.__config['audioDataPath'])

            # Loads the audio data(r) in binary format(b) into memory
            content = None
            with io.open(fileName, 'rb') as audioFile:
                content = audioFile.read()
            
            audio = speech.types.RecognitionAudio(content=content)

            # Set de config of the transcription.
            recognitionConfig = speech.types.RecognitionConfig(
                                    encoding=self.__config['encoding'],
                                    sample_rate_hertz=self.__config['sampleRate'],
                                    language_code=self.__config['languageCode'],
                                    model=self.__config['model'],
                                    use_enhanced=self.__config['enhanced'],
                                    enable_word_time_offsets=True)

            operation = client.long_running_recognize(recognitionConfig, audio)

            print('Waiting for operation to complete...')
            result = operation.result()

            for result in result.results:
                alternative = result.alternatives[0]
                srtFileName = self.__config['outputFolder'] + '/' + os.path.splitext(os.path.basename(self.__config['audioDataPath']))[0] + '.srt'
                with open(srtFileName, 'w') as file:
                    line = ''
                    lineStartTime = 0
                    lineEndTime = 0
                    ctr = 1
                    for wordInfo in alternative.words:
                        word = wordInfo.word
                        wordEndTime = wordInfo.end_time.seconds + wordInfo.end_time.nanos * 1e-9

                        tmpLine = line + ' ' + word

                        if len(tmpLine) < 70 and (wordEndTime - lineStartTime) < 6:
                            line = tmpLine
                            lineEndTime = wordEndTime
                        else:
                            file.write('{}\n'.format(ctr))
                            file.write('{:02d}:{:02d}:{:02d},{:03d} --> {:02d}:{:02d}:{:02d},{:03d}\n'.format(
                                int(lineStartTime // 360), int(lineStartTime / 60), int(lineStartTime % 60), int(1000 * (lineStartTime % 1)),
                                int(lineEndTime // 360), int(lineEndTime / 60), int(lineEndTime % 60), int(1000 * (lineEndTime % 1)),
                            ))

                            if len(line) <= 35:
                                file.write('{}\n'.format(line))
                            else:
                                firstPart = ''
                                secondPart = ''
                                firstPartIsNotFull = True
                                for lineWord in line.split(' '):
                                    if len((firstPart + ' ' + lineWord).lstrip(' ')) < 35 and firstPartIsNotFull:
                                        firstPart = firstPart + ' ' + lineWord
                                    else:
                                        firstPartIsNotFull = False
                                        secondPart = secondPart + ' ' + lineWord
                                file.write('{}\n{}\n'.format(firstPart.lstrip(' '), secondPart.lstrip(' ')))
                            file.write('\n')

                            # New line
                            ctr = ctr + 1
                            lineStartTime = wordInfo.start_time.seconds + wordInfo.start_time.nanos * 1e-9
                            lineEndTime = wordInfo.end_time.seconds + wordInfo.end_time.nanos * 1e-9
                            line = word

                    file.write('{}\n'.format(ctr+1))
                    file.write('{:02d}:{:02d}:{:02d},{:03d} --> {:02d}:{:02d}:{:02d},{:03d}\n'.format(
                        int(lineStartTime // 360), int(lineStartTime / 60), int(lineStartTime % 60), int(1000 * (lineStartTime % 1)),
                        int(lineEndTime // 360), int(lineEndTime / 60), int(lineEndTime % 60), int(1000 * (lineEndTime % 1)),
                    ))
                    if len(line) <= 35:
                        file.write('{}\n'.format(line))
                    else:
                        firstPart = ''
                        secondPart = ''
                        firstPartIsNotFull = True
                        for lineWord in line.split(' '):
                            if len((firstPart + ' ' + lineWord).lstrip(' ')) < 35 and firstPartIsNotFull:
                                firstPart = firstPart + ' ' + lineWord
                            else:
                                firstPartIsNotFull = False
                                secondPart = secondPart + ' ' + lineWord
                        file.write('{}\n{}\n'.format(firstPart.lstrip(' '), secondPart.lstrip(' ')))
                    file.write('\n')
                    
                self.transcriptionReady.emit(alternative.transcript)
        except Exception as e:
            self.exception.emit(e)


    # def __init__(self):
    #     pass


    # Synchronous Requests ~1 Minute

    # Asynchronous Requests ~480 Minutes

    # Streaming Requests ~1 Minutes