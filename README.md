# RendezVoUS [![Build Status](https://travis-ci.org/introlab/rendezvous.svg?branch=master)](https://travis-ci.org/introlab/rendezvous)

## Presentation

## Installation

1- Get the repository.

2- Go to the project working directory.
    
    $ cd rendezvous

3- Install Python 3.5, since it's the latest version supported on **Ubuntu 16.04**. We assume in the next steps that you are using **Ubuntu16.04**.

    $ sudo apt-get install python3

4 - Install pip for Python 3.5.

    $ sudo apt-get install python3-pip

5 - Install virtualenv for Python 3.5.

    $ python3 -m pip install --user virtualenv

6- Create a virtual environment of Python 3.5.

    $ python3 -m virtualenv env

7- Activate the virtualenv

    $ source ./env/bin/activate

- To deactivate the activated environment type : 

      $ deactivate

8- Install dependencies by running:

    $ sudo apt install vlc
    $ sudo apt-get install python3-tk
    $ sudo apt-get install xorg-dev libglu1-mesa-dev
    $ sudo apt-get install swig
    $ sudo apt-get install autoconf
    $ sudo apt-get install libtool
    $ pip install -r requirements.txt

9- Install audio processing libraries:

    $ git clone https://github.com/xiph/rnnoise
    $ cd rnnoise
    $ ./autogen.sh
    $ ./configure
    $ sudo make install

10- Generate the python ui with:

    $ python setup.py build_ui

11- build the c++ code and its dependencies use :

    $ make

- To only build the dewarping library alone use :

      $ make dewarping_lib

- To clean the c++ code and its dependencies use :

      $ make clean

- To only clean the dewarping library alone use :

      $ make clean_dewarping_lib

12 - If you add dependencies run the following command to add your new dependencies to requirements.txt:
    
    $ pip freeze > requirements.txt


12 - To use Yolo for face detection on the Jetson, run the following commands:

    $ chmod +x scripts/yolo_setup.sh
    $ ./scripts/yolo_setup.sh

## Testing
all unit tests are located in "tests" folder and the command to execute them is:

    $ cd rendezvous/ 
    $ pytest -v --cov=./ tests/

All tests files you add must follow this rule "test*.py", the framework use for testing is "unittest".

## Project Structure

- doc : for most documentation
- scripts : command-line scripts
- lib : C-language libraries
- src : application source code and tools related to the source code
- tests : unit testing
- config : for different config files
