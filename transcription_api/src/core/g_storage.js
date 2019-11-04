let {Storage} = require('@google-cloud/storage');

let GStorage = class {

    /**
     * Construct a GStorage, that allows you to play with buckets and blobs.
     * @param {string} serviceAccountPath 
     */
    constructor(serviceAccountPath) {
        this.serviceAccountPath = serviceAccountPath;
        this._storageClient = new Storage();
    }

    /**
     * Creates a bucket on google cloud storage.
     * @param {string} bucketName 
     */
    createBucket(bucketName) {
        this._storageClient.createBucket(bucketName);
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
        const blob = bucket.file(file.originalname);
        if (!blob) {
            next(new Error('cannot create blob'));
            return;
        }

        const blobStream = blob.createWriteStream();
        if (!blobStream) {
            next(new Error('unable to establish an upload stream'));
            return;
        }

        blobStream.on('error', function(err) {
            next(err);
        });

        blobStream.on('finish', function() {
            const fileUrl = format(`https://storage/googleapis.com/${bucketName}/${blob.name}`);
            next(null, fileUrl);
        });

        blobStream.end(file.buffer);
    }
};

module.exports = GStorage;
