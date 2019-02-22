# RendezVoUS

## Presentation

## Installation

1- Get the repository.

2- Go to the project working directory.
    
    $ cd rendezvous

3- Install Python 3.

    $ sudo apt-get install python3

4 - Install pip for Python 3.

    $ sudo apt-get install python3-pip

5 - Install virtualenv for Python 3.

    $ python3 -m pip install --user virtualenv

6- Create a virtual environment of Python 3.

    $ python3 -m virtualenv env

7- Activate the virtualenv

    $ source ./env/bin/activate

- To deactivate the activated environment type : 

      $ deactivate

8- Install dependencies by running:

    $ pip install -r requirements.txt

9 - If you add dependencies run the following command to add your dependency to requirements.txt:
    
    $ pip freeze > requirements.txt

## Project Structure

- doc : for most documentation
- scripts : command-line tools
- lib : C-language libraries
- src : application source code
- tests : unit testing
- config : for different config files
