import traceback

class ExceptionHelper:

    def __init__(self):
        pass


    @staticmethod
    def printStackTrace(exception):
        print('Exception : ', exception)
        traceback.print_tb(exception.__traceback__)