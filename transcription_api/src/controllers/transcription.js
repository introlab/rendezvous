let express = require('express');
let router = express.Router();

let httpErrors = require('../utils/HttpError');
let speech = require('../core/speech_to_text');
let GStorage = require('../core/g_storage');

router.get('/transcription', function(req, res, next) {
    // Request validation
    let uploadToGStorage = req.query.storage;
    

    // Request processing

    // Response
    res.sendStatus(httpErrors.OK);
});

module.exports = router;
