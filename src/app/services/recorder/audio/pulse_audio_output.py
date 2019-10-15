import ctypes
from src.app.services.recorder.audio.i_audio_output import IAudioOutput

pulseAudio = ctypes.cdll.LoadLibrary('libpulse-simple.so.0')

class struct_pa_sample_spec(ctypes.Structure):
    __slots__ = [
        'format',
        'rate',
        'channels',
    ]
struct_pa_sample_spec._fields_ = [
    ('format', ctypes.c_int),
    ('rate', ctypes.c_uint32),
    ('channels', ctypes.c_uint8),
]
pa_sample_spec = struct_pa_sample_spec

class PulseAudioOutput(IAudioOutput):
    def __init__(self, sinkName, sinkRate, sinkChannels, sinkFormat):
        self.name = sinkName

        self.ss = struct_pa_sample_spec()
        self.ss.rate = sinkRate
        self.ss.channels = sinkChannels
        self.ss.format = sinkFormat

        self.error = ctypes.c_int(0)

        self.stream = pulseAudio.pa_simple_new(
                            None,                    # Use default server.
                            'webrtc in',             # Application's name.
                            1,                       # Stream for playback, PA_STREAM_PLAYBACK = 1
                            self.name.encode(),      # PA device.
                            'playback',              # Stream's description.
                            ctypes.byref(self.ss),   # Sample format.
                            None,                    # Use default channel map.
                            None,                    # Use default buffering attributes.
                            ctypes.byref(self.error) # PA error code.
                        )

        if not self.stream:
            raise Exception('Could not create pulse audio stream! error: {0}'.format(self.error.value))

    def __del__(self):
        if pulseAudio.pa_simple_drain(self.stream, self.error):
            raise Exception('Could not drain data! error: {0}'.format(self.error.value))
        
        pulseAudio.pa_simple_free(self.stream)

    def getCurrentLatency(self):
        latency = pulseAudio.pa_simple_get_latency(self.stream, self.error)
        if latency == -1:
            raise Exception('Getting latency failed! error: {0}'.format(self.error.value))
        return latency

    def write(self, data):
        # Writing audio
        if pulseAudio.pa_simple_write(self.stream, data, len(data), self.error):
            raise Exception('Could not write to device! error: {0}'.format(self.error.value))