import io
import os
from enum import Enum, unique, auto

from google.cloud import speech


@unique
class EncodingType(Enum):
    ENCODING_UNSPECIFIED = 'ENCODING_UNSPECIFIED'
    FLAC = 'FLAC'
    AMR = 'AMR'
    AMR_WB = 'AMR_WB'
    LINEAR16 = 'LINEAR16'
    OGG_OPUS = 'OGG_OPUS'
    SPEEX_WITH_HEADER_BYTE = 'SPEEX_WITH_HEADER_BYTE'


@unique
class LanguageCode(Enum):
    FR_CA = 'fr-CA'
    EN_CA = 'en-CA'


@unique
class Model(Enum):
    DEFAULT = 'default'
    COMMAND_AND_SEARCH = 'command_and_search'
    PHONE_CALL = 'phone_call'
    VIDEO = 'video'


class SpeechToText:
    
        def __init__(self):
            pass


        @staticmethod
        def resquestTranscription(serviceAccountPath, audioDataPath, outputFolder,
            config={'encoding' : None, 'sampleRate' : 0, 'languageCode' : '', 'model' : '', 'enhanced' : False}):

            # Validations before starting transcription procedure.
            if not os.path.exists(serviceAccountPath):
                raise Exception('No Google Service Account File found at : {}'.format(serviceAccountPath))

            if not os.path.exists(audioDataPath):
                raise Exception('No Audio Data found at : {}'.format(audioDataPath))

            if not os.path.exists(outputFolder):
                raise Exception('No default output folder  found at : {}'.format(outputFolder))

            if not config['encoding'] in [encodingType.value for encodingType in EncodingType]:
                raise Exception('{} is not a supported encoding format'.format(config['encoding']))
            
            # Range accepted by the API.
            if not config['sampleRate'] in range(8000, 48001):
                raise Exception('Sample rate value {} is not in valid range'.format(config['sampleRate']))

            if not config['languageCode'] in [languageCode.value for languageCode in LanguageCode]:
                raise Exception('{} is not a supported language code'.format(config['languageCode']))
            
            if not config['model'] in [model.value for model in Model]:
                raise Exception('{} is not a supported model'.format(config['model']))

            # Instantiates a client.
            client = speech.SpeechClient.from_service_account_json(serviceAccountPath)

            # The name of the audio data to transcribe.
            fileName = os.path.join(audioDataPath)

            # Loads the audio data(r) in binary format(b) into memory
            content = None
            with io.open(fileName, 'rb') as audioFile:
                content = audioFile.read()
            
            audio = speech.types.RecognitionAudio(content=content)

            # Set de config of the transcription.
            recognitionConfig = speech.types.RecognitionConfig(
                                    encoding=config['encoding'],
                                    sample_rate_hertz=config['sampleRate'],
                                    language_code=config['languageCode'],
                                    model=config['model'],
                                    use_enhanced=config['enhanced'],
                                    enable_word_time_offsets=True)

            operation = client.long_running_recognize(recognitionConfig, audio)

            print('Waiting for operation to complete...')
            result = operation.result()

            for result in result.results:
                alternative = result.alternatives[0]
                srtFileName = outputFolder + '/' + os.path.splitext(os.path.basename(audioDataPath))[0] + '.srt'
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
                    
                return alternative.transcript

            # Detects speech in the audio data.
            #response = client.recognize(recognitionConfig, audio)

            # for result in response.results:
            #     # By default, the number of alternative is set to 1.
            #     transcription = result.alternatives[0].transcript
