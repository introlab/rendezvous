class FileParser:

    def __init__(self):
        pass

    @staticmethod
    def getLineFromFile(filePath, startsWith):
        with open(filePath, 'r') as fi:
            for line in fi:
                stripedLine = line.replace(' ', '')
                if stripedLine.startswith(startsWith):
                    return stripedLine