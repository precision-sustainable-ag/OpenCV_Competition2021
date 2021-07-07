
## Setting up the Raspberry Pi

### Install Python 3 on the RPi (???)


### Install the last version of DepthAI

`python3 -m pip install depthai==0.0.2.1+363f8cbae3af93e2b7283913dd82012e234641da --extra-index-url`

### Clone this branch

`git clone https://github.com/luxonis/depthai-python.git --branch gen2_develop`

### Calibrate the oak-d camera

Follow steps 2, 3, 4 and 5 on this document <https://docs.luxonis.com/tutorials/stereo_calibration/>
??? Check here that the document has changed. Which board should people use here, since they don't have the same camera configuration?
This calibration has to be done at the beginning of every week...


### Generate the calibration files which will contain the exposure and ISO specifications

On the terminal run:

`python3 00_control_exp_ISO_mono.py` 

Use x, x, x and x to to increase/ decrease the exp_time and ISO respectively. Once you are satisfied with the settings use the key s to save them. Press q to terminate the script. 

Next run:

`python3 01_control_exp_ISO_rgb.py`

Use x, x, x and x to to increase/ decrease the exp_time and ISO respectively. Use , . to control focus manually. Use w to cycle between white balance modes. Once you are satisfied with the settings use the key s to save them.Press q to terminate the script. 

### Collect rgb and depth images

On the terminal run:

`./run.sh`

The run.sh file has all the commands which will make the MM move in the positive direction (left to right) while the oak-d camera will collect images, and come back in the negative direction (right to left) making a video. The files will be saved on a folder...


### 00_control_exp_ISO_mono.py 

use i,o,k,l to increase/ decrease the exp_time and iso respectively. 
use s to save config into a file. 
it will be saved in pwd. 

### 01_control_exp_ISO_rgb.py

use i,o,k,l to increase/ decrease the exp_time and iso respectively. 
use , . to control focus manually 
use w to cycle between whitebalance modes 
use d to select EDOF mode for autofocus. 
use e to lock exposure 
use b to lock wb 
use s to save config into a file. 
it will be saved in pwd. 

### 02_save_rgb_depth_data.py 

Run script using.
02_save_rgb_depth_data.py -sp <path/to/save/images/> -n <num_of_images_to_save> -d <save depth,l,r,disp/rgb images> -c <use/calibration> -f <set_focus_mode> -mc <path/to/calibration_file_mono> -rc <path/to/calibration_file_rgb> 

At any point, use --help to get help on arguments to pass. 
Mono camera has no focus control, so in any case wait for 100 frames for the camera control parameters to kick in and then capture 'n' frames saved in 'sp' 

Will save the disparity maps and the same maps conv to depth arrays (.npy) in the specified directory. Modified from the gen2-camera-demo script on depthai-experiments. 

### 03_save_video.py 

same procedure as 02. 
videos saved as 720p encoded h264 (mono) and 4K encoded (h265) color in savepath. Need to convert to mp4 using ffmpeg. Commands to come up after pressing ctrl+c in the script. 

### 04_save_synced_frames.py 

Run script using.
04_save_synced_frames.py -sp <path/to/save/images/> -m <time_to_take_pictures> -d <dirty_flag(use if folder exists)> -c <use/calibration> -af <set_focus_mode_using_full_strings> -mc <path/to/calibration_file_mono> -rc <path/to/calibration_file_rgb> 

### 05_rgb_preview.py 

Run script using.
05_rgb_preview.py 
quit pressing "q" or ctrl+c into the terminal

use --help argument to go to help menu of any script. 
