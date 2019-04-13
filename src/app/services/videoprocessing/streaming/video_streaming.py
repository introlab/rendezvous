import multiprocessing
import queue
import time

from src.app.services.service.process.generic_process import GenericProcess
from src.utils.exception_helper import ExceptionHelper
from .video_stream import VideoStream

class VideoStreaming(GenericProcess):

    def __init__(self, cameraConfig):
        super(VideoStreaming, self).__init__()
        self.videoStream = VideoStream(cameraConfig)
        self.frameQueue = multiprocessing.Queue(1)


    def stop(self):
        super(VideoStreaming, self).stop()
        self.emptyQueue(self.frameQueue)


    def run(self):
        print('Starting video streaming')

        try:
            frame = None
            success = False

            self.videoStream.initializeStream()

            lastKeepAliveTimestamp = time.perf_counter()

            while not self.exit.is_set() and time.perf_counter() - lastKeepAliveTimestamp < 0.5:

                if frame is None:
                    success, frame = self.videoStream.readFrame()

                if success:
                    try:
                        self.frameQueue.put_nowait(frame)
                        frame = None
                    except queue.Full:
                        time.sleep(0.01)
                

                try:
                    self.keepAliveQueue.get_nowait()
                    lastKeepAliveTimestamp = time.perf_counter()
                except queue.Empty:
                    pass

        except Exception as e:
            ExceptionHelper.printStackTrace(e)
            self.exceptionQueue.put(e)

        finally:
            self.videoStream.destroy()
            print('Video streaming terminated')


    def getFrame(self, timeout):
        try:
            frame = self.frameQueue.get(timeout)
            return (True, frame)
        except queue.Empty:
            return (False, None)
