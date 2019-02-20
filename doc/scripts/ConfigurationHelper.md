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

To record ODAS tracking data use :

    python3 configuration_helper.py --cpath CONFIGPATH --opath ODASPATH

CONFIGPATH is the path to the ODAS configuration file to use and ODASPATH is the path to the odaslive program.

In addition you can change the number of ODAS events required before they are saved (default is 500) :

    --cs CHUNKSIZE

Finally, you can set how long you want the data gathering to last with this parameter (in minutes) :

    --time EXECUTIONTIME

2 - Analyse saved data

To analyse the previously saved data use :

    python3 configuration_helper.py --evalconf --srccpath SOURCECONFIGFPATH

Note the sources must be constantly emitting during the whole test and not move in order for the data to be relevent.