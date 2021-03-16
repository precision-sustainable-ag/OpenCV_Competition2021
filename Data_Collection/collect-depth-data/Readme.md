## Please add your code, notebooks, instructions, pipelines and reports.

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

