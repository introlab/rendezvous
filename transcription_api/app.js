let express = require('express');
let cookieParser = require('cookie-parser');
let logger = require('morgan');
let errorHandler = require('errorhandler');
let helmet = require('helmet');
let cors = require('cors');

require('./env');
let routes = require('./routes');

let app = express();
app.disable('x-powered-by');

// Third-party modules for our server.
if (process.env.NODE_ENV !== 'production') {
    app.use(logger('dev'));
    app.use(errorHandler({ dumpExceptions: true, showStack: true }));
}

app.use(helmet({
    frameguard: {
        action: 'deny'
    }
}));
app.disable('X-Powered-By');

app.use(cors({
    origin: process.env.ORIGIN
}));

app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());

//Connet all our routes to our application
routes(app);

module.exports = app;
