import os
import io

from google.cloud import speech

class SpeechToTextAPI:

        def __init__(self):
            pass


        @staticmethod
        def resquestTranscription(serviceAccountPath, audioDataPath,
            config={'encoding' : None, 'sampleRate' : 0, 'languageCode' : '', 'model' : '', 'enhanced' : False}):

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
                                    use_enhanced=config['enhanced'])

            # Detects speech in the audio data.
            response = client.recognize(recognitionConfig, audio)

            for result in response.results:
                # By default, the number of alternative is set to 1.
                transcription = result.alternatives[0].transcript

            return transcription
