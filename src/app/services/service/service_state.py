from enum import Enum, unique

@unique
class ServiceState(Enum):
    STARTING = 0,
    STOPPING = 1,
    READY = 2,
    RUNNING = 3,
    STOPPED = 4,
    TERMINATED = 5
    