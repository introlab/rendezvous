from google.cloud import storage


class GStorage():

    def __init__(self, serviceAccountPath):
        self.service_account_path = serviceAccountPath
    

    def createBucket(self, bucket_name):
        storageClient = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storageClient.create_bucket(bucket_name)


    def deleteBucket(self, bucket_name):
        storageClient = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storageClient.get_bucket(bucket_name)
        bucket.delete()


    def listBlobs(self, bucket_name):
        storageClient = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storageClient.get_bucket(bucket_name)
        blobs = bucket.list_blobs()


    def listBlobsWithPrefix(self, bucket_name, prefix, delimiter=None):
        storageClient = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storageClient.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)


    def uploadBlob(self, bucket_name, source_file_name, remoteFileName):
        storageClient = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storageClient.get_bucket(bucket_name)
        blob = bucket.blob(remoteFileName)
        blob.upload_from_filename(source_file_name)


    def downloadBlob(self, bucket_name, source_blob_name, destination_file_name):
        storageClient = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storageClient.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)


    def deleteBlob(self, bucket_name, blob_name):
        storageClient = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storageClient.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
    
