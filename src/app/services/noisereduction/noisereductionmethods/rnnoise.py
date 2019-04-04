# from github: https://gist.github.com/kazuki/7c7b50ba9df082bfe9f03819201648eb

from ctypes import byref, c_float, c_void_p, CDLL


class RNNoise(object):

    def __init__(self):
        self._frameSize = 480
        self._native = CDLL('/usr/local/lib/librnnoise.so')
        self._native.rnnoise_process_frame.restype = c_float
        self._native.rnnoise_process_frame.argtypes = (
            c_void_p, c_void_p, c_void_p)
        self._native.rnnoise_create.restype = c_void_p
        self._handle = self._native.rnnoise_create()
        self._buf = (c_float * self._frameSize)()
        

    def getFrameSize(self):
        return self._frameSize


    def processFrame(self, samples):
        if len(samples) > self._frameSize:
            raise ValueError
        for i in range(len(samples)):
            self._buf[i] = samples[i]
        for i in range(len(samples), self._frameSize):
            self._buf[i] = 0
        vad_prob = self._native.rnnoise_process_frame(
            self._handle, byref(self._buf), byref(self._buf))
        for i in range(len(samples)):
            samples[i] = self._buf[i]
        return vad_prob


    def __del__(self):
        if self._handle:
            self._native.rnnoise_destroy(self._handle)