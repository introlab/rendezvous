import os
import io

from google.cloud import speech

class SpeechToTextAPI:

        def __init__(self):
            pass

        @staticmethod
        def resquestTranscription(serviceAccountPath, audioDataPath, encoding, sampleRate, languageCode, model, enhanced):
            # Instantiates a client.
            client = speech.SpeechClient.from_service_account_json(serviceAccountPath)

            # The name of the audio data to transcribe.
            file_name = os.path.join(audioDataPath)

            # Loads the audio data into memory.
            with io.open(file_name, 'rb') as audio_file:
                content = audio_file.read()
            audio = speech.types.RecognitionAudio(content=content)

            # Set de config of the transcription.
            config = speech.types.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=sampleRate,
            language_code=languageCode,
            model=model,
            use_enhanced=enhanced)

            # Detects speech in the audio data.
            response = client.recognize(config, audio)

            for result in response.results:
                transcription = result.alternatives[0].transcript

            return transcription
