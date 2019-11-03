# Transcription API

this is the transcription api for the steno project. It communicates with googles servers to achieve speech-to-text transcription.

## Installation:

    $ cd ./transciption_api
    $ npm install

## Execution:

For development purpose:

    $ npm run start

For production:

    $ npm run prod

## Setup Https for localhost:

    $ cd ssl/
    
    $ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout cert.key -out cert.pem -config req.cnf -sha256

    $ cd ..
    $ npm run start

Hit https://localhost:3000 on your browser and accept the certificate.