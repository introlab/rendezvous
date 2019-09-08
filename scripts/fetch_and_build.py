import argparse
import sys
import os
import subprocess

class ArgsParser:

    def __init__(self):
        self.__parser = self.__createParser()
        self.args = self.__parser.parse_args()
        self.validateArgs()


    # arguments parser, all possible arguments are defined here. 
    def __createParser(self):
        parser = argparse.ArgumentParser(description='Helping with github branch fetcher')
        parser.add_argument('--branch', dest='branchName', action='store', help='Branch name to fetch')
        parser.add_argument('--commit', dest='commit', action='store', help='commit sha1 id to sync on.', default=None)
        return parser


    def validateArgs(self):
        if not self.args.branchName:
                raise self.__parser.error('--branch is required')


def gitFetchAndSync(branch, commit):
    command = 'git pull origin {b} && git checkout {b}'.format(b = branch)
    exitCode = os.system(command)
    if exitCode is not 0:
        raise Exception('git branch sync failed.')

    if (args.commit):
        command = 'git checkout {c}'.format(c = commit)
        exitCode = os.system(command)
        if exitCode is not 0:
            raise Exception('git commit sync failed.')

def buildPythonUI():
    command = 'python setup.py build_ui'
    exitCode = os.system(command)
    if exitCode is not 0:
        raise Exception('failed to build python UI')

def buildCppCode():
    command = 'make'
    exitCode = os.system(command)
    if exitCode is not 0:
        raise Exception('failed to build C++ code')

if __name__ == '__main__':
    parser = ArgsParser()
    args = parser.args

    backupBranch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).rstrip().decode("utf-8")
    backupCommit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).rstrip().decode("utf-8")

    try:
        gitFetchAndSync(args.branchName, args.commit)
        buildPythonUI()
        buildCppCode()
    except Exception as e:
        print(e)
        print('ROLLBACKING TO : ' + backupBranch + ' AT COMMIT ' + backupCommit)
        try:
            gitFetchAndSync(backupBranch, backupCommit)
            buildPythonUI()
            buildCppCode()
        except Exception as e:
            print(e)
            print('ROLLBACK FAILED')
            sys.exit(-1)
        print('COMMAND FAILED!!!!!! : ' + str(e))
        sys.exit(-1)  

    print('Everything is OK :-)')
    sys.exit(0)