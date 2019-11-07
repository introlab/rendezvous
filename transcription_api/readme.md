# Transcription API

this is the transcription api for the steno project. It communicates with Google's servers to achieve speech-to-text transcription.

## Installation:

    $ cd ./transciption_api
    $ npm install

## Execution:

For development purpose:

    $ npm run start

For production:

    $ npm run prod

## Setup Https for localhost and Client Certificates:

    $ cd ssl/
    
Certification Authority key and certificate:

    $ openssl req -new -x509 -days 365 -config ca.cnf -keyout ca-key.pem -out ca-crt.pem

Server's key generation:

    $ openssl genrsa -out server-key.pem 4096

Server's certification signature request:

    $ openssl req -new -config server.cnf -key server-key.pem -out server-csr.pem

Server's certificate authorization:

    $ openssl x509 -req -extfile server.cnf -days 365 -passin "pass:password" -in server-csr.pem -CA ca-crt.pem -CAkey ca-key.pem -CAcreateserial -out server-crt.pem

Client's key generation:

    $ openssl genrsa -out client.key 4096

Client's certificate generation:

    $ openssl req -new -config client.cnf -key client.key -out client.crt

Client's certificate approbation:

    $ openssl x509 -req -extfile client.cnf -days 365 -passin "pass:password" -in client.crt -CA ca-crt.pem -CAkey ca-key.pem -CAcreateserial -out client.crt

Verify the client's certificate is all right:

    $ openssl verify -CAfile ca-crt.pem client.crt

Hit https://localhost:3000 on your browser and accept the certificate.
