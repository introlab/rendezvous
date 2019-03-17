import json

class JsonUtils():

    @staticmethod
    def isJson(myJson):
        try:
            jsonObject = json.loads(myJson)
        except ValueError:
            return False
        return True
