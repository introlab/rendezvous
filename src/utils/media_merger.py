import os
import sys

class MediaMerger:

    def __init__(self):
        pass


    @staticmethod
    def fileMerger(audioInPath, videoInPath, mediaOutPath, srtInPath=None):
        isSrtValid = False
        
        
        # Input Audio Validation   
        if not audioInPath or (os.path.exists(audioInPath) == False):
            raise Exception('No file found at : {path}'.format(path=audioInPath))  
 
        # Input Video validation   
        if not videoInPath or (os.path.exists(videoInPath) == False):
            raise Exception('No file found at : {path}'.format(path=videoInPath))  
    
        # Srt file validation 
        if srtInPath:
            if(os.path.exists(srtInPath) == False):
                print('Invalid or no .srt file')  
            else:
                isSrtValid = True

        # Output file validation 
        if(os.path.splitext(mediaOutPath)[1] != '.avi'):
            raise Exception('Invalid output file format : {path} must be .avi'.format(path=mediaOutPath)) 

        # Creation of the ffmpeg command with the args
        if isSrtValid:
            ffmpegCommand = 'ffmpeg -y -i {} -i {} -vf subtitles={} {}'.format(audioInPath, videoInPath, srtInPath, mediaOutPath)
        else:
            ffmpegCommand = 'ffmpeg -y -i {} -i {} {}'.format(audioInPath, videoInPath, mediaOutPath)

        os.system(ffmpegCommand) 
        #print(ffmpegCommand)
