import argparse
from os import path

class ArgsParser:

    def __init__(self):
        self.__parser = self.__createParser()
        self.args = self.__parser.parse_args()
        self.validateArgs()

    # Arguments parser, all possible arguments are defined here. 
    def __createParser(self):
        parser = argparse.ArgumentParser(description='Transcribe audio data to text.')

        # See https://cloud.google.com/speech-to-text/docs/reference/rpc/google.cloud.speech.v1#recognitionconfig for detailed informations on all parameters.
        parser.add_argument('--audiopath', dest='audioPath', action='store', help='Path to the audio data.', required=True)
        parser.add_argument('--serviceaccountpath', dest='serviceAccountPath', action='store', help='Path to the google service account file (json).', required=True)
        parser.add_argument('--samplerate', dest='sampleRate', action='store', type=int, help='Sample rate of the audio data (Hz).', default=48000, required=False)
        parser.add_argument('--languagecode', dest='languageCode', action='store', type=str, help='Language of the audio data.', default='fr-CA', required=False)
        parser.add_argument('--maxalternatives', dest='maxAlternatives', action='store', type=int, help='Maximum number of alternatives returned.', default=1, required=False)
        parser.add_argument('--model', dest='model', action='store', type=str, help='Model used for transcription.', default='default', required=False)
        parser.add_argument('--useenhanced', dest='useEnhanced', action='store', type=bool, help='Use enhanced version on specified model.', default=True, required=False)
        parser.add_argument('--autopunctuation', dest='autoPunctuation', action='store', type=bool, help='Use automatic punctuation for transcription.', default=True, required=False)

        return parser


    def validateArgs(self):
        if not path.exists(self.args.audioPath):
            raise self.__parser.error('file at : {path} does not exists.'.format(path=self.args.audioPath))
        if not path.exists(self.args.serviceAccountPath):
            raise self.__parser.error('file at : {path} does not exists.'.format(path=self.args.serviceAccountPath))