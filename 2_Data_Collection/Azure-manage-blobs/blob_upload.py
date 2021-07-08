import os, uuid
import sys 
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
class DirectoryClient:
  def __init__(self, connection_string, container_name):
    service_client = BlobServiceClient.from_connection_string(connection_string)
    self.client = service_client.get_container_client(container_name)

  def upload(self, source, dest):
    '''
    Upload a file or directory to a path inside the container
    '''
    if (os.path.isdir(source)):
      self.upload_dir(source, dest)
    else:
      self.upload_file(source, dest)

  def upload_file(self, source, dest):
    '''
    Upload a single file to a path inside the container
    '''
    print(f'Uploading {source} to {dest}')
    with open(source, 'rb') as data:
      self.client.upload_blob(name=dest, data=data)

  def upload_dir(self, source, dest):
    '''
    Upload a directory to a path inside the container
    '''
    prefix = '' if dest == '' else dest + '/'
    prefix += os.path.basename(source) + '/'
    for root, dirs, files in os.walk(source):
      for name in files:
        dir_part = os.path.relpath(root, source)
        dir_part = '' if dir_part == '.' else dir_part + '/'
        file_path = os.path.join(root, name)
        blob_path = prefix + dir_part + name
        self.upload_file(file_path, blob_path)

connect_str = "BlobEndpoint=https://weedsmedia.blob.core.usgovcloudapi.net/;QueueEndpoint=https://weedsmedia.queue.core.usgovcloudapi.net/;FileEndpoint=https://weedsmedia.file.core.usgovcloudapi.net/;TableEndpoint=https://weedsmedia.table.core.usgovcloudapi.net/;SharedAccessSignature=sv=2020-02-10&ss=bfqt&srt=sco&sp=rwdlacupx&se=2029-03-01T18:44:49Z&st=2021-03-12T10:44:49Z&spr=https&sig=DCBFbtlUdA%2FQ%2F1COLUjnYYsogVuDuKx%2B622AxLoSqag%3D"#os.getenv('AZURE_STORAGE_CONNECTION_STRING')
container_name = "opencv"
blob_service_client = DirectoryClient(connect_str,container_name)



upload_log = "/home/pi/blob_upload_log.txt" 

directory = sys.argv[1]#main directory 
try:
    
    
   # Upload the directory
    blob_service_client.upload(directory,directory)
    print("\nUploading to Azure Storage as blob:\n\t" + directory)
    
        
    with open(upload_log, "a+") as file: 
        file.write("\n Succesfully uploaded file: \n" + directory)
    print("\n Succesfully uploaded file: \n"+directory)
except Exception as ex:
    print('Exception:')
    print(ex)
    with open(upload_log, "a+") as file: 
        file.write("\nFailed to upload file: ")
        file.write(directory+"\n")
        file.write("with the following exception: \n")
        file.write(str(ex)+"\n")
    pass

# num_of_args = len(sys.argv)
# for directory in sys.argv[1:]:  #main directory 
        # mainstreet = os.listdir(directory)
        # for subdirectory in range(len(mainstreet)):#subfolders
            # street = os.listdir(directory+'/'+mainstreet[subdirectory])
            # for image in range(len(street)): 
                # try:
                    # local_file = street[image] ####file to upload 
                    # local_file = directory+'/'+mainstreet[subdirectory]+'/'+local_file #join path+image
                    
                    # blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file)
                    # print("\nUploading to Azure Storage as blob:\n\t" + local_file)
                    # # Upload the created file
                    # with open(local_file, "rb") as data:
                        # blob_client.upload_blob(data)
                    # with open(upload_log, "a+") as file: 
                        # file.write("\n Succesfully uploaded file: \n" + local_file)
                    # print("\n Succesfully uploaded file: \n"+local_file)
                # except Exception as ex:
                    # print('Exception:')
                    # print(ex)
                    # with open(upload_log, "a+") as file: 
                        # file.write("\nFailed to upload file: ")
                        # file.write(local_file+"\n")
                        # file.write("with the following exception: \n")
                        # file.write(str(ex)+"\n")
                    # pass

