import context
import io
import os

from google.cloud import speech
from src.tools.speechtotext.argsparser.args_parser import ArgsParser

def main():
    try:
        # Get terminal arguments.
        parser = ArgsParser()
        args = parser.args

        print('speech_to_text starting... \n',
            'Audio data path : {} \n'.format(args.audioPath),
            'Service account path : {} \n'.format(args.serviceAccountPath),
            'Sampling rate : {} \n'.format(args.sampleRate),
            'Language code : {} \n'.format(args.languageCode),
            'Max alternatives : {} \n'.format(args.maxAlternatives),
            'Model : {} \n'.format(args.model),
            'Use enhanced : {} \n'.format(args.useEnhanced),
            'Automatic punctuation : {} \n'.format(args.autoPunctuation))
                
        # Instantiates a client.
        client = speech.SpeechClient.from_service_account_json(args.serviceAccountPath)
    
        # The name of the audio data to transcribe.
        file_name = os.path.join(args.audioPath)

        # Loads the audio data into memory.
        with io.open(file_name, 'rb') as audio_file:
            content = audio_file.read()
            audio = speech.types.RecognitionAudio(content=content)

        # Set de config of the transcription.
        config = speech.types.RecognitionConfig(
            sample_rate_hertz=args.sampleRate,
            language_code=args.languageCode,
            max_alternatives=args.maxAlternatives,
            model=args.model,
            use_enhanced=args.useEnhanced,
            enable_automatic_punctuation=args.autoPunctuation)

        # Detects speech in the audio data.
        response = client.recognize(config, audio)

        for result in response.results:
            for i in range(len(result.alternatives)):
                print('Transcript {alternative} : \n {result} \n'.format(
                alternative=i+1,
                result=result.alternatives[i].transcript))

    except Exception as e:
        print('Exception : ', e)
        raise e

if __name__ == '__main__':
    main()
