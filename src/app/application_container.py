
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from src.app.services.exceptions.exceptions import Exceptions
from src.app.services.odas.odas import Odas
from src.app.services.settings.settings import Settings
from src.app.services.videoprocessing.video_processor import VideoProcessor


class ApplicationContainer(containers.DeclarativeContainer):
    ''' Inversion of control(IoC) container of all the application object providers.'''

    exceptions = providers.Singleton(Exceptions)

    odas = providers.Singleton(Odas,
                               hostIP='127.0.0.1',
                               portPositions=10020, 
                               portAudio=10030, 
                               isVerbose=False)

    settings = providers.Singleton(Settings)

    videoProcessor = providers.Singleton(VideoProcessor)

