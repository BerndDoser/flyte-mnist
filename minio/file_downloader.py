from minio import Minio

client = Minio("minio-api-itssv197.h-its.org", secure=False)

bucket_name = "stream-test"

    
for item in client.list_objects(bucket_name, recursive=True):
    print(item.object_name)
    # client.fget_object(bucket_name, item.object_name, item.object_name)
