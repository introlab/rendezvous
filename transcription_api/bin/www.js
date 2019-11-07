#!/usr/bin/env node

/**
 * Module dependencies.
 */

let debug = require('debug')('transcription-api:server');
let https = require('https');
let fs = require('fs');

let app = require('../app');

/**
 * Get port from environment and store in Express.
 */

let port = normalizePort(process.env.PORT || '3000');
app.set('port', port);

let credentials = {
    key: fs.readFileSync('ssl/cert.key'),
    cert: fs.readFileSync('ssl/cert.pem')
};


/**
 * Create HTTPS server.
 */

let server = https.createServer(credentials, app);

/**
 * Listen on provided port, on all network interfaces.
 */

server.listen(port);
server.on('error', onError);
server.on('listening', onListening);

/**
 * Normalize a port into a number, string, or false.
 */

function normalizePort(val) {
    let port = parseInt(val, 10);

    if (isNaN(port)) {
        // named pipe
        return val;
    }

    if (port >= 0) {
        // port number
        return port;
    }

    return false;
}

/**
 * Event listener for HTTPS server "error" event.
 */

function onError(error) {
    if (error.syscall !== 'listen') {
        throw error;
    }

    let bind = typeof port === 'string'
        ? 'Pipe ' + port
        : 'Port ' + port;

    // handle specific listen errors with friendly messages
    switch (error.code) {
        case 'EACCES':
            console.error(bind + ' requires elevated privileges');
            process.exit(1);
            break;
        case 'EADDRINUSE':
            console.error(bind + ' is already in use');
            process.exit(1);
            break;
        default:
            throw error;
    }
}

/**
 * Event listener for HTTPS server "listening" event.
 */

function onListening() {
    let addr = server.address();
    let bind = typeof addr === 'string' ?
        'pipe ' + addr :
        'port ' + addr.port;
    debug('Listening on ' + bind);
}

/** Configuration time */
if (!process.env.ORIGIN || typeof process.env.ORIGIN !== 'string' ) {
    process.stdout.write('server error: bad origins.');
    process.exit();
}

process.stdout.write('server is ready...\n');