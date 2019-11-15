let {Storage} = require('@google-cloud/storage');

let GStorage = class {

    /**
     * Construct a GStorage, that allows you to play with buckets and blobs.
     */
    constructor() {
        this._storageClient = new Storage();
    }

    /**
     * Creates a bucket on google cloud storage.
     * @param {string} bucketName 
     */
    createBucket(bucketName, next) {
        this._storageClient.createBucket(bucketName, function(err) {
            next(err);
        });
    }

    /**
     * Verify that a specific bucket exist in Google cloud
     * @param {string} bucketName 
     * @param {function} next
     */
    bucketExist(bucketName, next) {
        const bucket = this._storageClient.bucket(bucketName);
        bucket.exists(function(err, exists) {
            return next(err, exists);
        });
    }

    /**
     * Upload a file on google cloud storage in the specified bucket.
     * @param {string} bucketName 
     * @param {File} file 
     * @param {function} next 
     */
    uploadFile(bucketName, file, next) {
        if (!file) {
            next(new Error('invalid file'));
            return;
        }

        let bucket = this._storageClient.bucket(bucketName);
        if (!bucket) {
            next(new Error('cannot create bucket'));
            return;
        }

        const blob = bucket.file(`${file.originalname}`);
        if (!blob) {
            next(new Error('cannot create blob'));
            return;
        }

        const type = file.mimetype;
        const blobStream = blob.createWriteStream({
            contentType: type,
            resumable: false
        });
        if (!blobStream) {
            next(new Error('unable to establish an upload stream'));
            return;
        }

        blobStream.on('error', function(err) {
            console.log(err.message);
            next(err);
        });

        blobStream.on('finish', function() {
            next(null);
        });

        blobStream.end(file.buffer);
    }
};

module.exports = GStorage;
