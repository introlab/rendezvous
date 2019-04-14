import io
import os
import time
import datetime
from enum import Enum, unique

from src.app.services.gstorage.g_storage import GStorage

from PyQt5.QtCore import QObject, QThread,  pyqtSignal, pyqtSlot

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
    __config = {
        'audioDataPath' : '',
        'encoding' : None,
        'enhanced' : False,
        'languageCode' : None,
        'model' : None,
        'outputFolder' : '',
        'sampleRate' : 0,
        'serviceAccountPath' : ''
    }

    # Valid range accepted by the Google API.
    __minSampleRate = 8000
    __maxSampleRate = 48000      
    # Value recommended for readability.
    __maxCharInSrtLine = 35
    # Value recommended for readability in second.
    __maxTimeForSrtBlock = 6


    def __init__(self, parent=None):
        super(SpeechToText, self).__init__(parent)

        # Worker thread so the transcription is not blocking.
        self.asynchroneSpeechToText = QThread()
        self.moveToThread(self.asynchroneSpeechToText)
        # What will run when the thread starts.
        self.asynchroneSpeechToText.started.connect(self.resquestTranscription)

        self.isRunning = False


    ''' SRT block format

        ID
        HH:MM:SS,mmm ---> HH:MM:SS,mmm
        Line 1
        Line 2
        Blank line

        Where H = Hour, M = Minute, S = Second, m = Millisecond
    '''
    def __getSrtBlock(self, id, startTime, endTime, text):
        # ID
        block = '{}\n'.format(id)
        
        # HH:MM:SS,mmm ---> HH:MM:SS,mmm    
        block += '{:02d}:{:02d}:{:02d},{:03d} --> {:02d}:{:02d}:{:02d},{:03d}\n'.format(
            int(startTime // 360), 
            int(startTime / 60), 
            int(startTime % 60), 
            int(1000 * (startTime % 1)),
            int(endTime // 360), 
            int(endTime / 60), 
            int(endTime % 60), 
            int(1000 * (endTime % 1))
        )

        # Might need to split the text in 2 lines.
        if len(text) <= self.__maxCharInSrtLine:
            # Line 1.
            block += '{}\n'.format(text)
        else:
            words = text.split(' ')
            tmpText = words[0]
            for idx, word in enumerate(words[1:]):
                preview = ' '.join([tmpText, word])
                if len(preview) > self.__maxCharInSrtLine:
                    # Saving remaining iteration by getting remaining words.
                    rest = '\n' + ' '.join(words[idx + 1:])
                    tmpText = ' '.join([tmpText, rest])
                    break
                else:
                    tmpText = preview
            # Line 1 and 2.
            block += '{}\n'.format(tmpText)

        # Blank line       
        block += '\n'

        return block


    def __getWordInfos(self, transcriptWord):
        word = transcriptWord.word
        wordStartTime = transcriptWord.start_time.seconds + transcriptWord.start_time.nanos * 1e-9 
        wordEndTime = transcriptWord.end_time.seconds + transcriptWord.end_time.nanos * 1e-9
        return [word, wordStartTime, wordEndTime]    


    def __generateSrtFile(self, fileName, transcriptWords):
        with open(fileName, 'w') as file:
            block, lineStartTime, lineEndTime = self.__getWordInfos(transcriptWords[0])
            id = 1
            for transcriptWord in transcriptWords[1:]:
                word, wordStartTime, wordEndTime = self.__getWordInfos(transcriptWord)
                tmpLine = (block + ' ' + word).strip()

                if len(tmpLine) < (self.__maxCharInSrtLine * 2) and (wordEndTime - lineStartTime) < self.__maxTimeForSrtBlock:
                    block = tmpLine
                    lineEndTime = wordEndTime
                else:
                    file.write(self.__getSrtBlock(id, lineStartTime, lineEndTime, block))

                    # New block.
                    id += 1
                    lineStartTime = wordStartTime
                    lineEndTime = wordEndTime
                    block = word

            file.write(self.__getSrtBlock(id, lineStartTime, lineEndTime, block))                
            

    def setConfig(self, config):
        self.__config = config


    def getMinSampleRate(self):
        return self.__minSampleRate


    def getMaxSampleRate(self):
        return self.__maxSampleRate


    def resquestTranscription(self):
        try:
            self.isRunning = True

            useGStorage = True

            # Validations before starting transcription procedure.
            serviceAccountPath = self.__config['serviceAccountPath']
            if not os.path.exists(serviceAccountPath):
                raise Exception('No Google Service Account File found at : {}'.format(serviceAccountPath))
            
            audioDataPath = self.__config['audioDataPath']
            if not os.path.exists(audioDataPath):
                raise Exception('No Audio Data found at : {}'.format(audioDataPath))

            outputFolder = self.__config['outputFolder']
            if not os.path.exists(outputFolder):
                raise Exception('No default output folder found at : {}'.format(outputFolder))

            encoding = self.__config['encoding']
            if not encoding in [encodingType.value for encodingType in EncodingTypes]:
                raise Exception('{} is not a supported encoding format'.format(encoding))
            
            sampleRate = self.__config['sampleRate']
            if not sampleRate in range(self.__minSampleRate, self.__maxSampleRate + 1):
                raise Exception('Sample rate value {} is not in valid range'.format(sampleRate))

            languageCode = self.__config['languageCode']
            if not languageCode in [languageCode.value for languageCode in LanguageCodes]:
                raise Exception('{} is not a supported language code'.format(languageCode))
            
            model = self.__config['model']
            if not model in [model.value for model in Models]:
                raise Exception('{} is not a supported model'.format(model))

            # Instantiates a client.
            client = speech.SpeechClient.from_service_account_json(serviceAccountPath)

            # The name of the audio data to transcribe.
            fileName = os.path.join(audioDataPath)

            if useGStorage:
                gstorage = GStorage(serviceAccountPath)

                st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
                bucketName = 'rdv-steno-{}'.format(st)
                remoteFileName = 'audio'

                gstorage.createBucket(bucketName)
                gstorage.uploadBlob(bucketName, fileName, remoteFileName)

                # gs://bucket-name/path_to_audio_file
                uri = 'gs://{}/{}'.format(bucketName, remoteFileName)
                audio = speech.types.RecognitionAudio(uri=uri)

            else :
                # Loads the audio data(r) in binary format(b) into memory.
                content = None
                with io.open(fileName, 'rb') as audioFile:
                    content = audioFile.read()

                audio = speech.types.RecognitionAudio(content=content)

            # Set de config of the transcription.
            recognitionConfig = speech.types.RecognitionConfig(
                                    encoding=encoding,
                                    sample_rate_hertz=sampleRate,
                                    language_code=languageCode,
                                    model=model,
                                    use_enhanced=self.__config['enhanced'],
                                    enable_word_time_offsets=True)

            operation = client.long_running_recognize(recognitionConfig, audio)

            result = operation.result()       

            totalWordsTranscription = []
            totalTranscription = ''
            for result in result.results:
                alternative = result.alternatives[0]
                totalTranscription = totalTranscription + alternative.transcript
                totalWordsTranscription.extend(alternative.words)

            self.transcriptionReady.emit(totalTranscription)
            # The SRT file name comes from the audio data file name.
            self.__generateSrtFile(fileName=outputFolder + '/' + os.path.splitext(os.path.basename(audioDataPath))[0] + '.srt',
                                       transcriptWords=totalWordsTranscription)

        except Exception as e:
            self.asynchroneSpeechToText.quit()
            self.exception.emit(e)

        finally:
            self.isRunning = False

