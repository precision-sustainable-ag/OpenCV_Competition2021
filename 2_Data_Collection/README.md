# General Data Collection folder

Here you can find three folders:

1 - DepthAI+RPi: All scripts on the RPi to manage the depthai framework. Those scripts were used to create the database with more than 3.00 vegetative structures during the competition.


2 - Azure-manage-blobs: Script and configuration for usisng Azure Cloud storage to submit all data we were getting in USDA-ARS Bestville - Maryland.


3 - Machine_Motion: Script to control the slider camera, assign different speed, acceleration and stop the camera to take images into the data collection process or even run the slider continously to record videos 


---

# Hardware Camera System

Data collection process used a customized camera version [DepthAI FFC](https://docs.luxonis.com/en/latest/pages/products/bw1098ffc/). With this camera version and with our our stereo baseline we could improve the minimun depth distance of the camera on the BenchBot top. This camera uses the same elements, but not integrated into a single board, has three FFC ports and has the advantage of changing the stereo module baseline to get shorter minimum depth distances.

We need to use DepthAI on our existing host (RPi). Since the AI/vision processing is done on the Myriad X, a typical desktop could handle tens of DepthAIs plugged in (the effective limit is how many USB ports the host can handle). 4K, 60Hz video camera with 12 MP stills and 4056 x 3040-pixel resolution. Depth + AI are needed, we need to use a modular, high-frame-rate, excellent-depth-quality cameras which can be separated to set up the stereo baseline.

For the complete deployment we used a Raspberry Pi 4 (4GB). This one is not only the host of DepthAI Camera, but also the controller of the Machine Motion Driver for lacoting the camera over the pots.

# Machine motion to control the slider camera:

XXXXX


