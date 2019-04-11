from google.cloud import storage

class GStorage():

    def __init__(self, serviceAccountPath):
        self.service_account_path = serviceAccountPath
    #exception = pyqtSignal(Exception)
    

    def create_bucket(self, bucket_name):
        storage_client = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storage_client.create_bucket(bucket_name)
        print('Bucket {} created'.format(bucket.name))

    def list_blobs(self, bucket_name):
        """Lists all the blobs in the bucket."""
        storage_client = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storage_client.get_bucket(bucket_name)

        blobs = bucket.list_blobs()
        i = 1
        print("Blob(s) found : ")
        for blob in blobs:
            print('{} - Blob {} found.'.format(i, blob.name))
            i += 1


    def list_blobs_with_prefix(self, bucket_name, prefix, delimiter=None):
        """Lists all the blobs in the bucket that begin with the prefix.
        This can be used to list all blobs in a "folder", e.g. "public/".
        The delimiter argument can be used to restrict the results to only the
        "files" in the given "folder". Without the delimiter, the entire tree under
        the prefix is returned. For example, given these blobs:
            /a/1.txt
            /a/b/2.txt
        If you just specify prefix = '/a', you'll get back:
            /a/1.txt
            /a/b/2.txt
        However, if you specify prefix='/a' and delimiter='/', you'll get back:
            /a/1.txt
        """
        storage_client = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storage_client.get_bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

        print('Blobs:')
        for blob in blobs:
            print(blob.name)

        if delimiter:
            print('Prefixes:')
            for prefix in blobs.prefixes:
                print(prefix)


    # [START storage_upload_file]
    def upload_blob(self, bucket_name, source_file_name):
        """Uploads a file to the bucket."""
        storage_client = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob("patate")

        blob.upload_from_filename(source_file_name)

        print('File {} uploaded to {}.'.format(source_file_name, "patate"))
    # [END storage_upload_file]


    def download_blob(self, bucket_name, source_blob_name, destination_file_name):
        """Downloads a blob from the bucket."""
        storage_client = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)

        print('Blob {} downloaded to {}.'.format(
            source_blob_name,
            destination_file_name))


    def delete_blob(self, bucket_name, blob_name):
        """Deletes a blob from the bucket."""
        storage_client = storage.Client.from_service_account_json(self.service_account_path)
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)

        blob.delete()

        print('Blob {} deleted.'.format(blob_name))
    
