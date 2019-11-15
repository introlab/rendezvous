let express = require('express');
let router = express.Router();
let multer = require('multer');
let async = require('async');

let httpErrors = require('../utils/HttpError');
let {SpeechToText} = require('../core/speech_to_text');
let speechToText = new SpeechToText();
let GStorage = require('../core/g_storage');

let multerMiddleware = multer({
    limits: {
        fieldSize: 25 * 1024 * 1024
    }
});

router.post('/transcription', multerMiddleware.single('audio'), function(req, res, next) {
    // Request validation
    let uploadToGStorage = req.query.storage == 'true';
    let bucketID = req.query.bucketID;
    let encoding = req.query.encoding;
    let enhanced = req.query.enhanced;
    let language = req.query.language;
    let sampleRate = req.query.sampleRate;
    let audioChannels = req.query.audioChannels;
    let model = req.query.model;
    let audio = req.file;

    if (!audio) {
        res.sendStatus(httpErrors.BAD_REQUEST);
        return;
    }

    if (uploadToGStorage && !bucketID) {
        res.sendStatus(httpErrors.BAD_REQUEST);
        return;
    }

    // Request processing
    async.waterfall([
        function(callback) {
            if (uploadToGStorage) {
                uploadAudioFile(bucketID, audio, callback);
                return;
            }
            callback(null);
        },
        function(callback) {
            let defaultConfig = speechToText.getConfig();

            let config = {
                audio: audio,
                encoding: encoding ? encoding : defaultConfig.encoding,
                enhanced: !!enhanced ? enhanced : defaultConfig.enhanced,
                languageCode: language ? language : defaultConfig.languageCode,
                model: model ? model : defaultConfig.model,
                sampleRate: sampleRate ? sampleRate : defaultConfig.sampleRate,
                audioChannelCount: audioChannels? audioChannels : defaultConfig.audioChannelCount
            };

            transcribe(config, callback);
        }
    ], function(err, transcription, words) {
        if (err) {
            res.status(httpErrors.INTERNAL_SERVER_ERROR).json({
                error: err.message
            });
            return next();
        }

        res.status(httpErrors.OK).json({
            transcription: transcription,
            words: words,
            error: err
        });
        return next();
    });
});

let uploadAudioFile = function(bucketName, audio, next) {
    let gstorage = new GStorage();

    async.waterfall([
        function(callback) {
            console.log("allo");
            gstorage.bucketExist(bucketName, function(err, exists) {
                if (err) {
                    return callback(err);
                }
                callback(null, exists);
            });
        },
        function(exists, callback) {
            if (exists) {
                return callback();
            }

            gstorage.createBucket(bucketName, function(err) {
                process.stdout.write('create bucket\n');
                return callback(err);
            });
        },
        function(callback) {
            gstorage.uploadFile(bucketName, audio, function(err) {
                if (err) {
                    return callback(err);
                }
                process.stdout.write(`upload done!\n`);
                callback(null);
            });
        }
    ], function(err) {
        next(err);
    });
};

let transcribe = function(config, next) {
    speechToText.setConfig(config);
    speechToText.requestTranscription(next);
};

module.exports = router;
