import json
import os


class FileHandler:

    def __init__(self):
        pass


    @staticmethod
    def getLineFromFile(filePath, startsWith):
        with open(filePath, 'r') as fi:
            for line in fi:
                stripedLine = line.replace(' ', '')
                if stripedLine.startswith(startsWith):
                    return stripedLine


    @staticmethod
    def writeJsonToFile(filePath, jsonData):
        if os.path.exists(filePath):
            os.remove(filePath)

        with open(filePath, mode='w', encoding='utf-8') as jsonFile:
            json.dump(jsonData, jsonFile)
