let speech = require('@google-cloud/speech');

const EncodingTypes = {
    ENCODING_UNSPECIFIED : 'ENCODING_USPECIFIED',
    FLAC : 'FLAC',
    AMR : 'AMR',
    AMR_WB : 'AMR_WB',
    LINEAR16 : 'LINEAR16',
    OGG_OPUS : 'OGG_OPUS',
    SPEEX_WITH_HEADER_BYTE : 'SPEEX_WITH_HEADER_BYTE'
};

/**
 * * Languages that we're using.
 */
const LanguageCodes = {
    FR_CA : 'fr-CA',
    EN_CA : 'en-CA'
};

/**
 * * AI models available with Google's speech-to-text service.
 */
const Models = {
    DEFAULT : 'default',
    COMMAND_AND_SEARCH : 'command_and_search',
    PHONE_CALL : 'phone_call',
    VIDEO : 'video'
};

/**
 * * Allows you to make request to Google's speech-to-text API.
 * * You can set different parameters, send pass your audio buffer and it returns a transcription.
 */
let SpeechToText = class {

    constructor() {
        this._client = new speech.SpeechClient();
        
        // ! All the parameters needed to perform a transcription.
        this._config = {
            audio : null,
            encoding : null,
            enhanced : false,
            languageCode : null,
            model : null,
            sampleRate : 0,
            audioChannelCount : 0,
        };

        // * Valid range accepted by the Google API
        this._minSampleRate = 8000;
        this._maxSampleRate = 48000;
        this._minChannelCount = 1;
        this._maxChannelCount = 254;
        // * Value recommended for readability
        this._maxCharInSrtLine = 35;
        // * Value recommended for readability in second.
        this._maxTimeForSrtBlock = 6;
    }

    setConfig(config) {
        if (!config) return;
        this._config = config;
    }

    /**
     * Minimal sample rate accepted by Speech API.
     */
    getMinSampleRate() {
        return this._minSampleRate;
    }

    /**
     * Maximal sample rate accepted by Speech API.
     */
    getMaxSampleRate() {
        return this._maxSampleRate;
    }

    /**
     * Minimal number of channels in the audio accepted by Speech API.
     */
    getMinChannelCount() {
        return this._minChannelCount;
    }

    /**
     * Maximal number of channels in the audio accepted by Speech API.
     */
    getMaxChannelCount() {
        return this._maxChannelCount;
    }

    /**
     * * Once the configuration is done, call this function to execute a transcription.
     * * It returns an error in next parameter if there is a problem with your configuration.
     * @param {function} next - Callback function.
     */
    async requestTranscription(next) {
        let error = this._validateInput();
        if (error) {
            next(error);
            return;
        }

        const audio = {
            content: this._config.audio
        };

        const config = {
            encoding: this._config.encoding,
            sampleRateHertz: this._config.sampleRate,
            audioChannelCount: this._config.audioChannelCount,
            languageCode: this._config.languageCode,
            model: this._config.model,
            useEnhanced: this._config.enhanced,
            enableWordTimeOffsets: true
        };

        const request = {
            audio: audio,
            config: config
        };

        const [response] = await this._client.recognize(request);
        const transcription = response.results
            .map(result => result.alternatives[0].transcript)
            .join('\n');

        console.log(transcription);
        next(null, transcription);
    }

    /**
     * Validates the configuration and returns an error if there is a problem.
     */
    _validateInput() {
        if (!this._config.audio) {
            return new Error('Invalid audio');
        }

        if (this._config.audio.length < 100) {
            return new Error('Audio file empty');
        }

        if (!EncodingTypes[this._config.encoding]) {
            return new Error(`${this._config.encoding} is not a supported encoding format.`);
        }

        if (this._config.sampleRate < this._minSampleRate || this._config.sampleRate > this._maxSampleRate) {
            return new Error(`Sample rate value ${this._config.sampleRate} is not in valid range.`);
        }

        if (this._config.audioChannelCount < this._minChannelCount || this._config.audioChannelCount > this._maxChannelCount) {
            return new Error(`Audio channel count value ${this._config.audioChannelCount} is not in valid range.`);
        }

        if (!Models[this._config.model]) {
            return new Error(`Model ${this._config.model} is not supported.`);
        }

        return null;
    }
}

module.exports = {
    SpeechToText: SpeechToText,
    Models: Models,
    LanguageCodes: LanguageCodes,
    EncodingTypes: EncodingTypes
};