import multiprocessing
import queue

class GenericProcess(multiprocessing.Process):

    def __init__(self):
        super(GenericProcess, self).__init__()
        self.keepAliveQueue = multiprocessing.Queue(1)
        self.exceptionQueue = multiprocessing.Queue()
        self.exit = multiprocessing.Event()


    def stop(self):
        self.exit.set()
        self.emptyQueue(self.keepAliveQueue)
        self.emptyQueue(self.exceptionQueue)


    def tryKeepAliveProcess(self):
        try:
            self.keepAliveQueue.put_nowait(True)
            return True
        except queue.Full:
            return False


    def tryRaiseProcessExceptions(self):
        try:
            raise Exception(self.__class__.__name__ + ' process exception : ' + str(self.exceptionQueue.get_nowait()))
        except queue.Empty:
            pass


    def emptyQueue(self, queue):
        try:
            while not queue.empty():
                queue.get()                
        except:
            pass