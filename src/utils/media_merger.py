
import os
import sys
import argparse

class MediaMerger:

    def __init__(self):
        pass

    @staticmethod
    def fileMerger(audioInPath, videoInPath, mediaOutPath, srtInPath=None):
        isSrtValid = False

        # Input Audio Validation 
        if(os.path.exists(audioInPath) == False):
            raise Exception('no file found at : {path}'.format(path=audioInPath))
            return
        # Input Video validation 
        if(os.path.exists(videoInPath) == False):
            raise Exception('no file found at : {}'.format(videoInPath)) 
            return
    
        # Srt file validation
        if srtInPath:
            if(os.path.exists(srtInPath) == False):
                print('Invalid or no .srt file')
                raise Exception('no file found at : {}'.format(srtInPath)) 
                #return
            else:
                isSrtValid = True

        # Output file validation 
        if(os.path.splitext(mediaOutPath)[1] != '.avi'):
            raise Exception('Invalid output file format : {path} must be .avi'.format(path=mediaOutPath)) 
            return

        if isSrtValid:
            ffmpegCommand = 'ffmpeg -y -i {} -i {} -vf subtitles={} {}'.format(audioInPath, videoInPath, srtInPath, mediaOutPath)
        else:
            ffmpegCommand = 'ffmpeg -y -i {} -i {} {}'.format(audioInPath, videoInPath, mediaOutPath)

        os.system(ffmpegCommand) 
        print(ffmpegCommand)