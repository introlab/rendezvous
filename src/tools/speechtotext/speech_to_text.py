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
            enable_automatic_punctuation=args.autoPunctuation,
            enable_word_time_offsets=True)

        # Detects speech in the audio data.
        # response = client.recognize(config, audio)

        # for result in response.results:
        #     for i in range(len(result.alternatives)):
        #         print('Transcript {alternative} : \n {result} \n'.format(
        #         alternative=i+1,
        #         result=result.alternatives[i].transcript))
        
        
        operation = client.long_running_recognize(config, audio)

        print('Waiting for operation to complete...')
        result = operation.result(timeout=90)

        for result in result.results:
            alternative = result.alternatives[0]
            print(u'Transcript: {}'.format(alternative.transcript))
            print('Confidence: {}'.format(alternative.confidence))

            with open('test.srt', 'w') as file:
                for ctr, word_info in enumerate(alternative.words):
                    word = word_info.word
                    start_time = word_info.start_time.seconds + word_info.start_time.nanos * 1e-9
                    end_time = word_info.end_time.seconds + word_info.end_time.nanos * 1e-9

                    file.write('{}\n'.format(ctr+1))
                    file.write('{:02d}:{:02d}:{:02d},{:03d} --> {:02d}:{:02d}:{:02d},{:03d}\n'.format(
                        int(start_time // 360), int(start_time / 60), int(start_time % 60), int(1000 * (start_time % 1)),
                        int(end_time // 360), int(end_time / 60), int(end_time % 60), int(1000 * (end_time % 1)),
                    ))
                    file.write('{}\n'.format(word))
                    file.write('\n')

                    # print('Word: {}, start_time: {}, end_time: {}'.format(
                    #     word,
                    #     start_time.seconds + start_time.nanos * 1e-9,
                    #     end_time.seconds + end_time.nanos * 1e-9))

    except Exception as e:
        print('Exception : ', e)
        raise e

if __name__ == '__main__':
    main()
