# Machine Motion (MM)
The BenchBot platform can be easily controlled using the MM controller which can be programmed using Python and the MM Python API. This makes the system highly versatile since we can integrate the MM Python API with other devices APIs.

## Setting up the Raspberry Pi
For our particular setup we are using a Raspberry Pi (RPi) 4 as our processing unit, through which we control the MM controller and the oak-d camera. 
Bellow are the requirements to set up the RPi to work with the MM.

### Install Python 3 on the RPi (???)

### Download the MM Python API

`git clone https://github.com/VentionCo/mm-python-api.git`

### Download the reqired libraries

`pip install -U socketIO-client`

`pip install -U pathlib`

`pip install -U paho-mqtt`

The RPi is now ready to work with the MM controller.

## MM scripts

Use any of the .py scripts included in this section to controll the MM


> ### **mm_control.py**
>
>> Use -d plus a distance in mm to move the camera to the right (positive direction). Use -hm to send the camera home (negative direction). This script is mostly meant to make videos since it moves the plate very slowly.
>>
>> ##### **Example** 
>>
>>`python3 mm_control.py -d 200`
>>
>> Will move the camera 200 mm in the positive direction


> ### **set_distance.py**
>
>> This script will ask how much do you want to move (in mm) and in which direction, positive (p) or negative (n). The speed at which the plate moves is higher than for the mm_cotrol.py script. Better suited for taking static pictures or for positioning the camera in a desired place. 
>>
>> ##### **On the terminal** 
>>
>>`python3 set_distance.py`


> ### **going_home.py**
>
>> This script sends the camera home at a fast speed.
>>
>> ##### **On the terminal** 
>>
>>`python3 going_home.py`
