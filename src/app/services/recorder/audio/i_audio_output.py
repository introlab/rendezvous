import abc

class IAudioOutput(abc.ABC):
    
    @abc.abstractmethod
    def write(self, data):
        pass