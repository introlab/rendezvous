import unittest
import pathlib
import os

from src.Utils.file_helper import FileHelper

rootPath = pathlib.Path('../../../').parents[2].absolute()


class TestFileHelper(unittest.TestCase):

    def setUp(self):
        self.testFile = os.path.join(rootPath, 'tests/Utils/FileHelper/data.json')
        pass


    # line doesn't exists
    def test_getLineFromFile_empty(self):
        line = FileHelper.getLineFromFile(self.testFile, 'abc')
        self.assertEqual(line, None)


    #line exists
    def test_getLineFromFile_valid(self):
        line = FileHelper.getLineFromFile(self.testFile, '"test"')
        print(line)
        self.assertEqual(line, '"test":"python",\n')


    # create file and write
    def test_writeJsonFile_not_exists(self):
        outputFile = os.path.join(rootPath, 'tests/Utils/FileHelper/test.json')
        jsonData = {
            'test' : 'rendezvous'
        }
        FileHelper.writeJsonFile(outputFile, jsonData)
        self.assertTrue(os.path.exists(outputFile))

        data = FileHelper.readJsonFile(outputFile)
        self.assertEqual(data, jsonData)
        os.remove(outputFile)


    # overwrite an existing file
    def text_writeJsonFile_exists(self):
        outputFile = os.path.join(rootPath, 'tests/Utils/FileHelper/test.json')
        oldData = {
            'test' : 'rendezvous'
        }
        FileHelper.writeJsonFile(outputFile, oldData)
        newData = {
            'test2' :  42
        }
        FileHelper.writeJsonFile(outputFile, newData)
        data = FileHelper.readJsonFile(outputFile)

        self.assertNotEqual(data, oldData)
        self.assertEqual(data, newData)
        os.remove(outputFile)

    # file doesn't exists
    def test_readJsonFile_not_exists(self):
        readFile = os.path.join(rootPath, 'tests/Utils/FileHelper/notExists.json')
        
        try:
            FileHelper.readJsonFile(readFile)
        
        except Exception:
            self.assertRaises(Exception)


    def test_readJsonFile_exists(self):
        readFile = os.path.join(rootPath, 'tests/Utils/FileHelper/test.json')
        data = {
            'test2' :  42
        }
        FileHelper.writeJsonFile(readFile, data)
        readedData = FileHelper.readJsonFile(readFile)

        self.assertEqual(data, readedData)
        os.remove(readFile)

    
    def test_readJsonFile_not_json(self):
        readFile = os.path.join(rootPath, 'tests/Utils/FileHelper/test.txt')
        data = 123423545435
        FileHelper.writeJsonFile(readFile, data)
        try: 
            FileHelper.readJsonFile(readFile)
        
        except Exception:
            self.assertRaises(Exception)
            os.remove(readFile)
