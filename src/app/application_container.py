
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from src.app.services.settings.settings import Settings


class ApplicationContainer(containers.DeclarativeContainer):
    ''' Inversion of control(IoC) container of all the application object providers.'''

    settings = providers.Singleton(Settings)

