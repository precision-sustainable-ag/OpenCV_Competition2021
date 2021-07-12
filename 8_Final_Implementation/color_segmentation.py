#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import sys
import joblib 
import statsmodels.api as sm  
#dependancies: 
#pip3 install joblib
#pip3 install scikit-learn==0.24.1
#pip3 install matplotlib
#sudo apt-get install python3-gi-cairo
'''
Deeplabv3 person running on selected camera.
Run as:
python3 -m pip install -r requirements.txt
python3 deeplabv3_person_256.py -cam rgb
Possible input choices (-cam):
'rgb', 'left', 'right'

Blob taken from the great PINTO zoo

git clone git@github.com:PINTO0309/PINTO_model_zoo.git
cd PINTO_model_zoo/026_mobile-deeplabv3-plus/01_float32/
./download.sh
source /opt/intel/openvino/bin/setupvars.sh
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py   --input_model deeplab_v3_plus_mnv2_decoder_256.pb   --model_name deeplab_v3_plus_mnv2_decoder_256   --input_shape [1,256,256,3]   --data_type FP16   --output_dir openvino/256x256/FP16 --mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5]
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -ip U8 -VPU_NUMBER_OF_SHAVES 6 -VPU_NUMBER_OF_CMX_SLICES 6 -m openvino/256x256/FP16/deeplab_v3_plus_mnv2_decoder_256.xml -o deeplabv3p_person_6_shaves.blob

'''

import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from collections import deque

def make_fv(x,depth): #RGB, DepthFrame 
    # rgb_img: np array in [RGB] channel order
    # EXG = 2 * G - R - B
    img = x
    #width = int(img.shape[1])
    #height = int(img.shape[0])
    #dsize = (width, height)
    #mask = cv2.resize(x['mask'], dsize)
    #img[:, :, 0:2][mask==0] = 0
    blue = img[:,:,2]
    green = img[:,:,1]
    red = img[:,:,0]
    exg = 2*green - red - blue
    exg = np.where(exg < 0, 0, exg).astype('uint8') # Thresholding removes low negative values (noise)
    exr = 1.4 * red - green
    exr = np.where(exr < 0, 0, exr).astype('uint8') # Thresholding removes low negative values
    
    try:
      m_exg, sd_exg = cv2.meanStdDev(exg)
    except:
      m_exg, sd_exg = 0
      pass
    all_exg = exg.sum() 
    
    try:
      m_exr, sd_exr = cv2.meanStdDev(exr)
    except:
      m_exr, sd_exr = 0 
      pass
    all_exr=exr.sum()
    
    NUM_BINS = 20
    depth_histogram = np.histogram(depth,list(range(0,65535,50)))
    #print(depth_histogram)
    histogram = np.zeros((len(depth_histogram),NUM_BINS))
    #for i in range(0,len(depth_histogram)):
       # histogram[i,:] = depth_histogram[i][0][0:NUM_BINS]
    histogram[:] = depth_histogram[0][0:NUM_BINS]
    features = 20 
    sum_depth = histogram.sum()
    #sum_depth = np.zeros(len(histogram))
    #for i in range(len(histogram)):
       # sum_depth[i] = sum(histogram[i][0:features])
    return all_exg, m_exg, sd_exg, all_exr, m_exr, sd_exr, sum_depth 


class DataPlot:
    def __init__(self, max_entries=20):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
        self.axis_y2 = deque(maxlen=max_entries)
        self.axis_y3 = deque(maxlen=max_entries) 
        self.max_entries = max_entries

        self.buf1 = deque(maxlen=5)
        self.buf2 = deque(maxlen=5)

    def add(self, x, y,y2,y3):
        self.axis_x.append(x)
        self.axis_y.append(y)
        self.axis_y2.append(y2)
        self.axis_y3.append(y3)


class RealtimePlot:
    def __init__(self, axes):
        self.axes = axes

        self.lineplot, = axes[0].plot([], [], "yo-")
        self.lineplot2, = axes[1].plot([], [],"ro-")
        self.lineplot3, = axes[2].plot([], [],"mo-")
        self.axes[0].set_title("Grass Biomass")
        self.axes[1].set_title("Clover Biomass")
        self.axes[2].set_title("Broadleaf Biomass")
        self.axes[0].set_ylabel("Grams")
        self.axes[1].set_ylabel("Grams")
        self.axes[2].set_ylabel("Grams")
    def plot(self, dataPlot):
        self.lineplot.set_data(dataPlot.axis_x, dataPlot.axis_y)
        self.lineplot2.set_data(dataPlot.axis_x, dataPlot.axis_y2)
        self.lineplot3.set_data(dataPlot.axis_x, dataPlot.axis_y3)
        
        self.axes[0].set_xlim(min(dataPlot.axis_x), max(dataPlot.axis_x))
        ymin = 0
        ymax = max(dataPlot.axis_y) + 1
        self.axes[0].set_ylim(ymin, ymax)
        self.axes[0].relim();
    
        
        self.axes[1].set_xlim(min(dataPlot.axis_x), max(dataPlot.axis_x))
        ymin = 0
        ymax = max(dataPlot.axis_y2) + 1
        self.axes[1].set_ylim(ymin, ymax)
        self.axes[1].relim();
        
        self.axes[2].set_xlim(min(dataPlot.axis_x), max(dataPlot.axis_x))
        ymin = 0
        ymax = max(dataPlot.axis_y3) + 1
        self.axes[2].set_ylim(ymin, ymax)
        self.axes[2].relim();


cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)#'models/deeplab_v3_plus_mvn2_decoder_256_openvino_2021.2_6shave.blob'
parser.add_argument("-nn", "--nn_model", help="select camera input source for inference", default='models/4_class_model_mobilenet_v3_large_data4_combined_class_weights_512x512_without_softmax.blob', type=str)
#old model: class_model_mobilenet_v3_small_data3_class_weights_512x512_without_softmax_6shaves.blob
#
args = parser.parse_args()

cam_source = args.cam_input 
nn_path = args.nn_model 

nn_shape = 256 #size of square image 
if '513' in nn_path:
    nn_shape = 513
if '512' in nn_path:
    nn_shape = 512
def decode_deeplabv3p(output_tensor):
    #["Soil":BROWN,"Clover:RED","Broadleaf:PURPLE","Grass:ORANGE"]
    class_colors = [[40, 86,166 ], [28, 26,228 ], [184 , 126, 155], [0, 127, 255]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    
    output = output_tensor.reshape(nn_shape,nn_shape)
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame):
    #save output colors. 
    
    return cv2.addWeighted(frame,1, output_colors,0.2,0)



# Start defining a pipeline
pipeline = dai.Pipeline()

if '513' in nn_path:
    nn_shape = 513
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
if '512' in nn_path:
    nn_shape = 512
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_3)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam=None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(nn_shape,nn_shape)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam.preview.link(detection_nn.input)
elif cam_source == 'left':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
elif cam_source == 'right':
    cam = pipeline.createMonoCamera()
    cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

if cam_source != 'rgb':
    manip = pipeline.createImageManip()
    manip.setResize(nn_shape,nn_shape)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

#cam.setFps(40)not used in PSA implementation 
####################ROI##########################################
stepSize = 0.01
newConfig = False
# Define a source - two mono (grayscale) cameras
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
#spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutDepth = pipeline.createXLinkOut()
#xoutSpatialData = pipeline.createXLinkOut()
#xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutDepth.setStreamName("depth")
#xoutSpatialData.setStreamName("spatialData")
#xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# MonoCamera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

outputDepth = True
outputRectified = True
lrcheck = True
subpixel = False

# StereoDepth 
stereo.setOutputDepth(outputDepth)
stereo.setOutputRectified(outputRectified)
stereo.setConfidenceThreshold(255)
#stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_7x7)#Artem
#stereo.setExtendedDisparity(True)


stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
####################Depth Cropping##################
# Crop range
topLeftDepth = dai.Point2f(0, 0) ## X1/720,Y2/1280
bottomRightDepth = dai.Point2f(1, 1) ##X2/720 Y2/1280 
#Properties 
manip = pipeline.createImageManip()
manip.initialConfig.setCropRect(topLeftDepth.x, topLeftDepth.y, bottomRightDepth.x, bottomRightDepth.y)
manip.setMaxOutputFrameSize(monoRight.getResolutionHeight()*monoRight.getResolutionWidth()*3)
#Linking: 
configIn = pipeline.createXLinkIn()
configIn.setStreamName('config')
configIn.out.link(manip.inputConfig)
stereo.depth.link(manip.inputImage)
manip.out.link(xoutDepth.input)
###########################ROI#########################
#spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
#stereo.depth.link(spatialLocationCalculator.inputDepth)
#
#increaseROI = 0
#topLeft = dai.Point2f(0.5-increaseROI, 0.5-increaseROI)
#bottomRight = dai.Point2f(0.6+increaseROI, 0.6+increaseROI)
#
#spatialLocationCalculator.setWaitForConfigInput(False)
#config = dai.SpatialLocationCalculatorConfigData()
#config.depthThresholds.lowerThreshold = 100
#config.depthThresholds.upperThreshold = 10000
#config.roi = dai.Rect(topLeft, bottomRight)
#spatialLocationCalculator.initialConfig.addROI(config)
#spatialLocationCalculator.out.link(xoutSpatialData.input)
#xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

#############################################
# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
device = dai.Device(pipeline)
device.startPipeline()

# Output queues will be used to get the rgb frames and nn data from the outputs defined above
q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False) #Nueral network input
q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)#Neural network output 
# Output queue will be used to get the depth frames from the outputs defined above
############################ROI###################################################
depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
#spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
#spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
####################################################################################
start_time = time.time()
counter = 0
fps = 0
layer_info_printed = False
###PLOT DEPTH DATA#############
fig, axes = plt.subplots(3,1)
data = DataPlot();
dataPlotting = RealtimePlot(axes)
count = 0 
fig.tight_layout()
#plt.title('Real-Time BioMass Estimation',pad = 220)
##Load BioMass Estimation Models; 
broadl_rf = joblib.load("./models/random_forest_broadl.joblib")
clover_rf = joblib.load("./models/random_forest_clover.joblib")
grass_rf = joblib.load("./models/random_forest_grass.joblib")
clover_sm = sm.load('./models/model_broadl_stats.pickle')
broadl_sm = sm.load('./models/model_clover_stats.pickle')
grass_sm = sm.load('./models/model_grass_stats.pickle')
while True:
    count +=1
    ####################################ROI##################################
    inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived

    depthFrame = inDepth.getFrame()#npy array, 720x1280 
    #print(depthFrame.shape)
    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
    depthFrameColor = cv2.equalizeHist(depthFrameColor)
    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
    
    #spatialData = spatialCalcQueue.get().getSpatialLocations()
#    for depthData in spatialData: ROI plotting 
#        roi = depthData.config.roi
#        roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
#        xmin = int(roi.topLeft().x)
#        ymin = int(roi.topLeft().y)
#        xmax = int(roi.bottomRight().x)
#        ymax = int(roi.bottomRight().y)
#
#        fontType = cv2.FONT_HERSHEY_TRIPLEX
#        color = (255, 255, 255)
#        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
#        cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, 255)
#        cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, 255)
#        cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, 255)
    #Plot Depth: NEED to implement with .npy instead of spatialCoordinates
    
    # Show the frame
    cv2.imshow("depth", depthFrameColor)
    #Move ROI
    newConfig = False
    key = cv2.waitKey(1) & 0xFF # Artem
    if key == ord('q'):
        break
    # elif key == ord('w'):
        # if topLeft.y - stepSize >= 0:
            # topLeft.y -= stepSize
            # bottomRight.y -= stepSize
            # newConfig = True
    # elif key == ord('a'):
        # if topLeft.x - stepSize >= 0:
            # topLeft.x -= stepSize
            # bottomRight.x -= stepSize
            # newConfig = True
    # elif key == ord('s'):
        # if bottomRight.y + stepSize <= 1:
            # topLeft.y += stepSize
            # bottomRight.y += stepSize
            # newConfig = True
    # elif key == ord('d'):
        # if bottomRight.x + stepSize <= 1:
            # topLeft.x += stepSize
            # bottomRight.x += stepSize
            # newConfig = True
    # elif key == ord('e'):
        # topLeft.x += 0.01
        # topLeft.y += 0.01
        # bottomRight.x -= 0.01
        # bottomRight.y -= 0.01
        # newConfig = True
    # elif key == ord('r'):
        # topLeft.x -= 0.01
        # topLeft.y -= 0.01
        # bottomRight.x += 0.01
        # bottomRight.y += 0.01
        # newConfig = True
    # if newConfig:
        # config.roi = dai.Rect(topLeft, bottomRight)
        # cfg = dai.SpatialLocationCalculatorConfig()
        # cfg.addROI(config)
        # spatialCalcConfigInQueue.send(cfg)
        # newConfig = False

    #######################################ROI#############################################
    # instead of get (blocking) used tryGet (nonblocking) which will return the available data or None otherwise
    in_nn_input = q_nn_input.get()
    in_nn = q_nn.get()

    if in_nn_input is not None: ########Neural Network input 
        # if the data from the rgb camera is available, transform the 1D data into a HxWxC frame
        shape = (3, in_nn_input.getHeight(), in_nn_input.getWidth())
        
        frame = in_nn_input.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        #print("RGB input shape:", frame.shape)
    if in_nn is not None: ######Nueral Network Prediciton 
        # print("NN received")
        layers = in_nn.getAllLayers()

        if not layer_info_printed:
            for layer_nr, layer in enumerate(layers):
                print(f"Layer {layer_nr}")
                print(f"Name: {layer.name}")
                print(f"Order: {layer.order}")
                print(f"dataType: {layer.dataType}")
                dims = layer.dims[::-1] # reverse dimensions
                print(f"dims: {dims}")
            layer_info_printed = True

        # get layer1 data
        layer1 = in_nn.getLayerInt32(layers[0].name)
        # reshape to numpy array
        dims = layer.dims[::-1]
        lay1 = np.asarray(layer1, dtype=np.int32).reshape(dims)
        
        output_colors = decode_deeplabv3p(lay1)
        #############Finding max/min points: 
        clover_indices = np.where(lay1[0] == 1) #get indices where segmask values are present. 
        broadl_indices = np.where(lay1[0]==2)
        grass_indices = np.where(lay1[0]==3)
        #Feature vector:['all_exg','m_exg','sd_exg', 'all_exr', 'm_exr', 'sd_exr', 'all_sumbinary','sum_depth'
        
        try: 
          g_Xmax = np.amax(grass_indices[1])
          g_Xmin =  np.amin(grass_indices[1])
          g_Ymax =  np.amax(grass_indices[0])
          g_Ymin = np.amin(grass_indices[0])
          g_depth = depthFrame[g_Ymin+60:g_Ymax+60,g_Xmin+115:g_Xmax+115] #Paula's transformation
          g_all_sumbinary = len(grass_indices[0]) #Count of pixels in segmask 
          g_exg, g_m_exg, g_sd_exg, g_exr, g_m_exr, g_sd_exr, g_sumdepth  = make_fv(frame[g_Ymin:g_Ymax,g_Xmin:g_Xmax],g_depth)
          #Now make feature vector and biomass predictions with random forest. 
          g_vec = np.asarray([g_exg, g_m_exg[0][0], g_sd_exg[0][0], g_exr, g_m_exr[0][0], g_sd_exr[0][0],g_all_sumbinary, g_sumdepth])
          g_vec = g_vec.reshape(1,-1)#must reshape if only one sample
          grass_bm = grass_rf.predict(g_vec)
          #print("Grass Biomass: ",grass_bm)
         
          c_Xmax = np.amax(clover_indices[1])
          c_Xmin =  np.amin(clover_indices[1])
          c_Ymax =  np.amax(clover_indices[0])
          c_Ymin = np.amin(clover_indices[0])
          c_depth = depthFrame[c_Ymin+60:c_Ymax+60,c_Xmin+115:c_Xmax+115]
          c_all_sumbinary = len(clover_indices[0]) #Count of pixels in segmask 
          c_exg, c_m_exg, c_sd_exg, c_exr, c_m_exr, c_sd_exr, c_sumdepth  = make_fv(frame[c_Ymin:c_Ymax,c_Xmin:c_Xmax],c_depth)
          #Now make feature vector and biomass predictions with random forest. 
          c_vec = np.asarray([c_exg, c_m_exg[0][0], c_sd_exg[0][0], c_exr, c_m_exr[0][0], c_sd_exr[0][0],c_all_sumbinary, c_sumdepth])
          c_vec = c_vec.reshape(1,-1)#must reshape if only one sample
          clover_bm = clover_rf.predict(c_vec)
          #print("Clover Biomass: ", clover_bm)
          
          b_Xmax = np.amax(broadl_indices[1])
          b_Xmin =  np.amin(broadl_indices[1])
          b_Ymax =  np.amax(broadl_indices[0])
          b_Ymin = np.amin(broadl_indices[0])
          b_depth = depthFrame[b_Ymin+60:b_Ymax+60,b_Xmin+115:b_Xmax+115]
          b_all_sumbinary = len(broadl_indices[0]) #Count of pixels in segmask 
          b_exg, b_m_exg, b_sd_exg, b_exr, b_m_exr, b_sd_exr, b_sumdepth  = make_fv(frame[b_Ymin:b_Ymax,b_Xmin:b_Xmax],b_depth)
          #Now make feature vector and biomass predictions with random forest. 
          b_vec = np.asarray([b_exg, b_m_exg[0][0], b_sd_exg[0][0], b_exr, b_m_exr[0][0], b_sd_exr[0][0],b_all_sumbinary, b_sumdepth])
          b_vec = b_vec.reshape(1,-1)#must reshape if only one sample
          broadl_bm = broadl_rf.predict(b_vec)
          #print("Broadleaf Biomass: ", broadl_bm)
          
          data.add(count, grass_bm[0],clover_bm[0],broadl_bm[0]) #Plots grass biomass
          dataPlotting.plot(data)
          plt.pause(0.001)
          
        except ValueError as e: #empty array 
          #print(e)
          pass
       
          
        labeled_output = output_colors #New screen for segmentation mask + labels 
        cv2.putText(labeled_output, "Soil: BROWN, Clover: RED, Broadleaf: PURPLE , Grass: ORANGE", (2, labeled_output.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.45, (255, 0, 0))
        cv2.imshow("seg_mask",labeled_output)
        if frame is not None:
            frame = show_deeplabv3p(output_colors, frame)
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
            try:
              frame = cv2.rectangle(frame,(c_Xmin,c_Ymin),(c_Xmax,c_Ymax),(255,0,0),2)
            except:
              pass
            cv2.imshow("nn_input", frame)#was frame
            #cv2.imshow("seg_mask",output_colors)
    
    counter+=1
    if (time.time() - start_time) > 1 :
        fps = counter / (time.time() - start_time)

        counter = 0
        start_time = time.time()

