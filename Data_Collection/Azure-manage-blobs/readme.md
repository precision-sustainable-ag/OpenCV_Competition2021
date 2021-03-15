### Instalation instructions

Run this command line in your terminal

```bash
!python3 -m pip install -r requirements.txt
```

### Upload process

```bash
python3 blob_upload.py <path1> <path2> <path3> <path4> ... <or_particular_image/video>
```

Where <path> is the directory of images/videos you would like to upload. Use this format.

```bash
python3 blob_upload.py \
    /home/pi/depthai-python/data_timestamp \
```

### Notes

The program will go through each directory and upload files that are present within it. If the file already exists in Azure, no duplicate files are uploaded. The program expects the directories to only have files, and no other directories. If this is difficult to guarantee we could only upload certain file types. 

A log file called blob_output_log.txt is created, and looks like this after a successful upload, and then an attempted duplicate upload: 

![](https://lh6.googleusercontent.com/mX1Rdnr-H_8ZeL1jN3cZhbW40mJ-Q3ojyEHkPTaYqTj-X7gMduoklOU91o5TY02-_pTuJl58tmW2sD1LpY1CjaXBRHICyyKDHet0xnCkq8lM-fbEMyZaWb_EU8vgCAxqE2yMyUDd)

Expected usage: 
```bash
python blob_upload.py directory1 directory2 directory3 directory directory5 ...  image.png
```

```bash
Succesfully uploaded file:
/home/pi/images/25_07_2020_19_57_00.jpg
Succesfully uploaded file:
/home/pi/images/03_08_2020_07_17_39.jpg
Succesfully uploaded file:
/home/pi/images2/23_07_2020_12_27_02.jpg
Succesfully uploaded file:
/home/pi/images2/25_07_2020_17_52_51.jpg
Failed to upload file: /home/pi/images/25_07_2020_19_57_00.jpg
with the following exception:
The specified blob already exists.
RequestId:1b95c4da-501e-00a3-59d0-196a5b000000
Time:2021-03-15T19:19:08.5418478Z
ErrorCode:BlobAlreadyExists
Error:None

Failed to upload file: /home/pi/images/03_08_2020_07_17_39.jpg
with the following exception:
The specified blob already exists.
RequestId:1b95cd67-501e-00a3-03d0-196a5b000000
Time:2021-03-15T19:19:14.6039315Z
ErrorCode:BlobAlreadyExists
Error:None

Failed to upload file: /home/pi/images2/23_07_2020_12_27_02.jpg
with the following exception:
The specified blob already exists.
RequestId:1b95d389-501e-00a3-34d0-196a5b000000
Time:2021-03-15T19:19:21.8177890Z
ErrorCode:BlobAlreadyExists
Error:None

Failed to upload file: /home/pi/images2/25_07_2020_17_52_51.jpg
with the following exception:
The specified blob already exists.
RequestId:1b95dd79-501e-00a3-70d0-196a5b000000
Time:2021-03-15T19:19:29.9472629Z
ErrorCode:BlobAlreadyExists
Error:None
```
