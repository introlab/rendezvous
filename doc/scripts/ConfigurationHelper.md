# Configuration Helper

## Configuration

1 - ODAS configuration file

Pass the configuration file as an argument using :

    --cpath CONFIGFILE

The ODAS config files are situated in config/testcnfigs/ODAS

2 - Source location configuration file

Pass the configuration file as an argument using :

    --srccpath CONFIGFILE

The source location config files are situated in config/testcnfigs/sources

## Usage

1 - Record data with the microphone matrice

TODO

2 - Analyse saved data

To analyse the previously saved data use :

    python3 configuration_helper.py --evalconf --srccpath CONFIGFILE

Note the sources must be constantly emitting during the whole test and not move in order for the data to be relevent.