

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

echo "Taking data. Position 2"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 3"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 4"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 5"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 6"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 7"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 8"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz


echo "Taking data. Position 9"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 10"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 11"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 12"

python3 /home/pi/one_third_relative_move.py

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

#python3 video_relative_move.py

#python3 03_save_video.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
# -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz
#Added by Artem, please verify. Changing the sleep interval may be necessary
python3 video_relative_move.py &
sleep 1s && python3 03_save_video.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \ -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz &
