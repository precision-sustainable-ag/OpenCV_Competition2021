import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

try:
    #print("Azure Blob Storage v" + __version__ + " - Python quickstart sample")
    # Retrieve the connection string for use with the application. The storage
    # connection string is stored in an environment variable on the machine
    # running the application called AZURE_STORAGE_CONNECTION_STRING. If the environment variable is
    # created after the application is launched in a console or with Visual Studio,
    # the shell or application needs to be closed and reloaded to take the
    # environment variable into account.
    connect_str = "BlobEndpoint=https://weedsmedia.blob.core.usgovcloudapi.net/;QueueEndpoint=https://weedsmedia.queue.core.usgovcloudapi.net/;FileEndpoint=https://weedsmedia.file.core.usgovcloudapi.net/;TableEndpoint=https://weedsmedia.table.core.usgovcloudapi.net/;SharedAccessSignature=sv=2020-02-10&ss=bfqt&srt=sco&sp=rwdlacupx&se=2029-03-01T18:44:49Z&st=2021-03-12T10:44:49Z&spr=https&sig=DCBFbtlUdA%2FQ%2F1COLUjnYYsogVuDuKx%2B622AxLoSqag%3D"#os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_name = "opencv"
    
    local_file = "/home/pi/25_07_2020_17_52_51.jpg" ####File to upload 
    
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file)
    print("\nUploading to Azure Storage as blob:\n\t" + local_file)
    # Upload the created file
    with open(local_file, "rb") as data:
        blob_client.upload_blob(data)

except Exception as ex:
    print('Exception:')
    print(ex)


