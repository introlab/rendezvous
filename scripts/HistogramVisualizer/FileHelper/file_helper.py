import json
import os


class FileHelper:

    def __init__(self):
        pass


    @staticmethod
    def getLineFromFile(filePath, startsWith):
        if not os.path.exists(filePath):
            raise Exception('no file found at : {path}'.format(path=filePath))

        with open(filePath, 'r') as fi:
            for line in fi:
                stripedLine = line.replace(' ', '')
                if stripedLine.startswith(startsWith):
                    return stripedLine


    @staticmethod
    def writeJsonFile(filePath, jsonData):
        if os.path.exists(filePath):
            os.remove(filePath)

        with open(filePath, mode='w', encoding='utf-8') as jsonFile:
            json.dump(jsonData, jsonFile)


    @staticmethod
    def readJsonFile(filePath):
        if not os.path.exists(filePath):
            raise Exception('no file found at : {path}'.format(path=filePath))
        
        if not filePath.endswith('.json'):
            raise Exception('not a json file')

        with open(filePath, mode='r', encoding='utf-8') as jsonFile:
            data = json.load(jsonFile)
            return data