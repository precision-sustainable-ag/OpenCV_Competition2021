

#!/bin/bash

# file: run.sh
#
# This script takes a picture Full Size and Video and Collect synchronized depth data
#


# take a picture
# take a picture
echo "Taking data. Don't move the camera."
date #was |& tee -a

#echo "First check that the three pot columns are inside of the image"
#echo "Please press "q" to START with the line protocol"

#count=0
#while true; do
#	read -n 1 k <&1
#	if [[ $k = q ]] 
#	then
#		printf "\nQuitting from the program\n"
#		break
#	else
#		printf "\nIterate for times\n"
#		python3 01_rgb_preview.py
#		echo "Press 'q' to exit"
#	fi
#done

#python3 02_save_rgb_depth_data.py 

echo "Taking data. Position 1"

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") -n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 2"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 3"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 4"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 5"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 6"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 7"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 8"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz


echo "Taking data. Position 9"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 10"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 11"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 12"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 13"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 14"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 15"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 16"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

echo "Taking data. Position 17"

python3 /home/pi/mm_control.py -d 61.25

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d r -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 02_save_rgb_depth_data.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
-n 30 -d d -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 video_relative_move.py

python3 03_save_video.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") \
 -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz

python3 /home/pi/mm_control.py -hm &
sleep 1s && python3 03_save_video.py -sp /home/pi/collected_data/week1/$(date +"%Y_%m_%d_%H_%M_%S") -c use_calibration -f 3 -mc /home/pi/mono_calib.npz -rc /home/pi/RGB_calib.npz &


