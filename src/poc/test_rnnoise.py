import sys
import numpy as np

from rnnoise import RNNoise

FRAME_SIZE = 480

# main
rnnoise = RNNoise()
denoisedDataArray = np.empty([], 'int16')

with open(sys.argv[1], 'rb') as soundFile:
    soundDataArray = np.fromfile(soundFile, np.int16)


for i in range(0, len(soundDataArray), FRAME_SIZE):
    tmp = soundDataArray[i:i+FRAME_SIZE]
    rnnoise.process_frame(tmp)
    denoisedDataArray = np.append(denoisedDataArray, tmp)

fout = open('/home/walid/projet/sandbox/denoisedTest2.raw', 'w')
denoisedDataArray.tofile(fout)
fout.close()

