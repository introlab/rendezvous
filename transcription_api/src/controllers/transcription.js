let express = require('express');
let router = express.Router();
let multer = require('multer');
let async = require('async');

let httpErrors = require('../utils/HttpError');
let {SpeechToText} = require('../core/speech_to_text');
let GStorage = require('../core/g_storage');

router.get('/transcription', multer().single('audio'), function(req, res, next) {
    // Request validation
    let uploadToGStorage = req.query.storage;
    let encoding = req.query.encoding;
    let enhanced = req.query.enhanced;
    let language = req.query.language;
    let model = req.query.model;
    let sampleRate = req.query.sampleRate;
    let audioChanels = req.query.audioChanels;
    let audio = req.file;

    if (!audio) {
        res.sendStatus(httpErrors.BAD_REQUEST);
        return;
    }

    // Request processing
    let uploadedFileUrl = '';

    async.waterfall([
        function(callback) {
            if (uploadToGStorage) {
                uploadAudioFile(audio, callback);
                return;
            }
            callback(null, null);
        },
        function(fileUrl, callback) {
            uploadedFileUrl = fileUrl;
            
            let config = {
                audio: audio,
                encoding: encoding ? encoding : defaultConfig.encoding,
                enhanced: !!enhanced ? enhanced : defaultConfig.enhanced,
                languageCode: language ? language : defaultConfig.languageCode,
                model: model ? model : defaultConfig.model,
                sampleRate: sampleRate ? sampleRate : defaultConfig.sampleRate,
                audioChannelCount: audioChanels? audioChanels : defaultConfig.audioChannelCount
            };

            transcribe(config, callback);
        }
    ], function(err, transcription) {
        if (err) {
            res.status(httpErrors.INTERNAL_SERVER_ERROR).json({
                error: err.message
            });
            return next();
        }

        res.status(httpErrors.OK).json({
            transcription: transcription,
            fileUrl: uploadedFileUrl,
            error: err
        });
        return next();
    });
});

let uploadAudioFile = function(audio, next) {
    let gstorage = new GStorage();
    let date = new Date();
    let st = `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}:${date.getHours()}:${date.getMinutes()}:${date.getSeconds()}`;
    let bucketName = `rdv-steno-${st}`;
    gstorage.uploadFile(bucketName, audio, function(err, fileUrl) {
        if (err) {
            next(err);
            return;
        }
        next(null, fileUrl);
    });
};

let transcribe = function(config, next) {
    let speechToText = new SpeechToText();
    let defaultConfig = speechToText.getConfig();

    speechToText.setConfig(config);
    speechToText.requestTranscription(next);
};

module.exports = router;
