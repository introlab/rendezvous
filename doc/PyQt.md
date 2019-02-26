# PyQt

## Linux installation

### QtCreator

1 - Download the open source installer at: https://www.qt.io/download

2 - Run the following command on the downloaded file:

    $ chmod +x qt-unified-linux-x64-3.0.6-online.run

3 - Run the installer:

    $ ./qt-unified-linux-x64-3.0.6-online.run

4 - During installation, choose the following version of Qt: 5.12.1

5 - Install build essentials:

    $ sudo apt-get install build-essential

6 - Install packages:

    $ sudo apt-get install mesa-common-dev
    $ sudo apt-get install libglu1-mesa-dev -y
    $ sudo apt-get install pyqt5-dev-tools

For more information:

    https://wiki.qt.io/Install_Qt_5_on_Ubuntu


## Usage

### Modifying the UI

Open QtCreator and modify the file: prototype/gui/mainwindow.ui

Generate the python ui with:

    $ cd src
    $ python setup.py build_ui


### Running the app

Install the PyQt dependency (only the first time)

    $ pip install

Run the app:

    $ python prototype/main.py --opath "path to odaslive" --cpath "path to config file"
