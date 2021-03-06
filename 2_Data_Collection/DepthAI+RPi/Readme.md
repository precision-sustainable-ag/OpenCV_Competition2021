
## Data collection steps

### Install the last version of DepthAI

`python3 -m pip install depthai`

### Clone this branch

`git clone https://github.com/luxonis/depthai-python.git --branch gen2_develop`

### Calibrate the oak-d camera

Follow [this document](<https://docs.luxonis.com/tutorials/stereo_calibration/>) to calibrate the camera. For our particular set up we used a custom board according to the [DepthAI FFC](https://docs.luxonis.com/en/latest/pages/products/bw1098ffc/) camera version. The custom json board calibration file can be seen [here](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/blob/master/2_Data_Collection/DepthAI%2BRPi/WEED01.json
). This json file needs to be included in this path /home/pi/<depthai_path>/resources/boards.


The camera calibration was done every week to guarantee that any change on the dimensions of the plastica case due to temperature variations was taking into account. 


### Generate the calibration files which will contain the exposure and ISO specifications

On the terminal run:

`python3 00_control_exp_ISO_mono.py` 

Use i,o,k and l to to increase/ decrease the exp_time and ISO respectively. Once you are satisfied with the settings use the key s to save them. Press q to terminate the script. 

Next run:

`python3 01_control_exp_ISO_rgb.py`

Use i,o,k and l to to increase/ decrease the exp_time and ISO respectively. Use , . to control focus manually. Use w to cycle between white balance modes. Once you are satisfied with the settings use the key s to save them. Press q to terminate the script. 

### Collect rgb and depth images

On the terminal run:

`./run.sh`

The run.sh file has all the commands which will make the MM move in the positive direction (left to right) while the oak-d camera will collect images, and come back in the negative direction (right to left) making a video. The files will be saved on a folder named according to the metadata: date and time, pot line number on the bench and stop number of the camera slider within the line. More details about the protocol [here](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/wiki/2.-Data-Collection
).

### Upload the files to Azure

Before the upload make sure that you follow [this steps](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/blob/master/2_Data_Collection/Azure-manage-blobs/readme.md) to make sure all the requirementas are satisfied.

Copy the path to the folder that needs to be uploaded. 

On the terminal run:

`python3 blob_upload.py + folder path`

## DepthAI scripts

> ### **00_control_exp_ISO_mono.py**
> Change and save exposure time and ISO settings, for the mono camera. Creates a file with the settings called **mono_calib.npz** which will be later used for the image collection.
>
>> Use i,o,k,l to increase/ decrease the exp_time and ISO respectively.
>>
>> Use s to save the desired settings into a .npz file.
>>
>> Use q or ctrl+c to terminate.


> ### **01_control_exp_ISO_rgb.py**
> Change and save exposure time and ISO settings, control focus manually and more, for the rgb camera. Creates a file with the settings called **RGB_calib.npz** which will be later used for the image collection. 
>
>> Use i,o,k,l to increase/ decrease the exp_time and ISO respectively.
>>
>> Use , . to control focus manually.
>>
>> Use w to cycle between whitebalance modes.
>>
>> Use d to select EDOF mode for autofocus.
>>
>> Use e to lock exposure.
>>
>> Use b to lock wb 
>>
>> Use s to save the desired settings into a .npz file.
>>
>> Use q or ctrl+c to terminate.

> ### **02_save_rgb_depth_data.py**
> Save rgb and depth data on a specified folder.
>
>> This script takes the arguments -sp (path to where images will be saved), -n (number of images to be saved), -d (save depth,l,r,disp/rgb images), -c (use calibration), -f (focus mode, xx to xx), -mc (path to mono calibration file), -rc (path to RGB calibration file).
>>
>> **Example**
>>
>> `python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$number/stop11/$(date +"%Y_%m_%d_%H_%M_%S") -n 30 -d -dr r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz`
>>
>> Mono camera has no focus control, so in any case wait for 100 frames for the camera control parameters to kick in and then capture 'n' frames saved in 'sp'.
>>
>> Will save the disparity maps and the same maps conv to depth arrays (.npy) in the specified directory. Modified from the gen2-camera-demo script on depthai-experiments. 

> ### **03_save_video.py**
> Save videos as 720p encoded h264 (mono) and 4K encoded h265 (color) on a specified path.
>
>> This script takes the arguments -sp (path to where images will be saved), -c (use calibration), -f (focus mode, xx to xx), -mc (path to mono calibration file), -rc (path to RGB calibration file).
>>
>> **Example**
>>
>> `python3 03_save_video.py -sp /home/pi/collected_data/week1/$number/videos/$(date +"%Y_%m_%d_%H_%M_%S") -c use_calibration -f 5 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz`
>>

> ### 04_save_synced_frames.py 
>
>> Run script using.
>> 04_save_synced_frames.py -sp <path/to/save/images/> -m <time_to_take_pictures> -d <dirty_flag(use if folder exists)> -c <use/calibration> -af <set_focus_mode_using_full_strings> -mc <path/to/calibration_file_mono> -rc <path/to/calibration_file_rgb> 


> ### **05_rgb_preview.py**
> Visualize what the oak-d camera is seeing. Opens pop-up windows for the rgb and depth cameras.
>
>> Quit by clicking on one of the windows and pressing q or ctrl+c in the terminal.

use --help argument to go to help menu of any script. 

For more details visit our [wiki](https://github.com/precision-sustainable-ag/OpenCV_Competition2021/wiki/2.-Data-Collection) page
