# PyQt

## Linux installation

### QtCreator

1 - Download the open source installer at: https://www.qt.io/download

2 - Run the following command on the downloaded file:

    $ chmod +x qt-unified-linux-x64-3.0.6-online.run

3 - Run the installer:

    $ ./qt-unified-linux-x64-3.0.6-online.run

4 - During installation, choose the following version of Qt: 5.12.1

For more information:

    https://wiki.qt.io/Install_Qt_5_on_Ubuntu


## Usage

### Modifying the UI

Open QtCreator and modify the ui files: app/gui/*.ui

Generate the python ui with:

    $ python setup.py build_ui


### Running the app

Install the PyQt dependency (only the first time)

    $ pip install -r requirements.txt

Run the app:

    $ python src/app/main.py
