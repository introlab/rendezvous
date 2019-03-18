import json

class JsonUtils():

    @staticmethod
    def isJson(myJson):
        try:
            jsonObject = json.loads(str(myJson, 'utf-8'))
        except Exception:
            return False
        return True
