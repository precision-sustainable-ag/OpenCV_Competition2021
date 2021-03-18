

#!/bin/bash

# file: run.sh
#
# This script takes a picture Full Size and Video and Collect synchronized depth data
#


# take a picture
# take a picture
echo "Taking data. Don't move the camera."
date #was |& tee -a

#python3 /home/pi/to_first_pot_relative_move.py

echo "Taking data. Position 1"

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz




