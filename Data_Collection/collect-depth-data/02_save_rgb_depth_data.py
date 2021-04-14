#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from time import sleep
import datetime
import argparse
import os
from pathlib import Path
'''
If one or more of the additional depth modes (lrcheck, extended, subpixel)
are enabled, then:
 - depth output is FP16. TODO enable U16.
 - median filtering is disabled on device. TODO enable.
 - with subpixel, either depth or disparity has valid data.

Otherwise, depth output is U16 (mm) and median is functional.
But like on Gen1, either depth or disparity has valid data. TODO enable both.
'''
parser = argparse.ArgumentParser()
parser.add_argument("-sp", "--savepath", default='/home/pi/collected_depth/', help="output path for depth data")
parser.add_argument("-n", "--numframes", default=5, help="Number of frames to be saved")
parser.add_argument('-d', '--dirty', action='store_true', default=False, help="Allow the destination path not to be empty")
parser.add_argument("-dr", "--depthrgb", default='d', help="Capture Depth images (d) or RGB (r)")
parser.add_argument("-c","--usecalibration", default=True, help="Use calibrated parameters for 3A for both cameras")
parser.add_argument("-f","--focusMode", default=5, help="Focus Modes. Enter exactly from [0:'MANUAL',1:'AUTO',3:'CONTINUOUS_VIDEO',4:'CONTINUOUS_PICTURE',5:'EDOF']")
parser.add_argument("-mc", "--calibrationfilemono", default='/home/pi/mono_calib.npz')
parser.add_argument("-rc", "--calibrationfilergb", default='/home/pi/RGB_calib.npz')
args = parser.parse_args()
#if not os.path.exists(args.savepath):
#    os.makedirs(args.savepath)

dest = Path(args.savepath).resolve().absolute()
if dest.exists() and len(list(dest.glob('*'))) != 0 and not args.dirty:
    raise ValueError(f"Path {dest} contains {len(list(dest.glob('*')))} files. Either specify new path or use \"--dirty\" flag to use current one")
dest.mkdir(parents=True, exist_ok=True)

n=args.numframes
sp=args.savepath
collecting=args.depthrgb
use_calibration=args.usecalibration
if use_calibration:
    calib_mono=np.load(args.calibrationfilemono)
    calib_color=np.load(args.calibrationfilergb)
focus_mode=args.focusMode
# StereoDepth config options. TODO move to command line options
source_camera  = not False
out_depth      = False  # Disparity by default
out_rectified  = True   # Output and display rectified streams
lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# Sanitize some incompatible options
if lrcheck or extended or subpixel:
    median   = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF # TODO

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)
# TODO add API to read this from device / calib data
right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]

def clamp(num, v0, v1): return max(v0, min(num, v1))

def create_rgb_cam_pipeline():
    print("Creating pipeline: RGB CAM -> XLINK OUT")
    pipeline = dai.Pipeline()

    cam          = pipeline.createColorCamera()
    control_in = pipeline.createXLinkIn()
    control_in.setStreamName('control_r')
    xout_preview = pipeline.createXLinkOut()
    xout_video   = pipeline.createXLinkOut()


    cam.setPreviewSize(540, 540)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    xout_preview.setStreamName('rgb_preview')
    xout_video  .setStreamName('rgb_video')

    control_in.out.link(cam.inputControl)
    cam.preview   .link(xout_preview.input)
    cam.video     .link(xout_video.input)

    streams = ['rgb_preview', 'rgb_video']

    return pipeline, streams

def create_stereo_depth_pipeline(from_camera=True):
    print("Creating Stereo Depth pipeline: ", end='')
    if from_camera:
        print("MONO CAMS -> STEREO -> XLINK OUT")
    else:
        print("XLINK IN -> STEREO -> XLINK OUT")
    pipeline = dai.Pipeline()

    if from_camera:
        cam_left      = pipeline.createMonoCamera()
        cam_right     = pipeline.createMonoCamera()
    else:
        cam_left      = pipeline.createXLinkIn()
        cam_right     = pipeline.createXLinkIn()

    control_in = pipeline.createXLinkIn()
    control_in.setStreamName('control_m')
    control_in.out.link(cam_left.inputControl)
    control_in.out.link(cam_right.inputControl)

    stereo            = pipeline.createStereoDepth()
    xout_left         = pipeline.createXLinkOut()
    xout_right        = pipeline.createXLinkOut()
    xout_depth        = pipeline.createXLinkOut()
    xout_disparity    = pipeline.createXLinkOut()
    xout_rectif_left  = pipeline.createXLinkOut()
    xout_rectif_right = pipeline.createXLinkOut()
    if from_camera:
        cam_left .setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        for cam in [cam_left, cam_right]: # Common config
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
            #cam.setFps(20.0)
    else:
        cam_left .setStreamName('in_left')
        cam_right.setStreamName('in_right')

    stereo.setOutputDepth(out_depth)
    stereo.setOutputRectified(out_rectified)
    stereo.setConfidenceThreshold(255)
    stereo.setRectifyEdgeFillColor(0) # Black, to better see the cutout
    stereo.setMedianFilter(median) # KERNEL_7x7 default
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)
    if from_camera:
        # Default: EEPROM calib is used, and resolution taken from MonoCamera nodes
        #stereo.loadCalibrationFile(path)
        pass
    else:
        stereo.setEmptyCalibration() # Set if the input frames are already rectified
        stereo.setInputResolution(1280, 720)

    xout_left        .setStreamName('left')
    xout_right       .setStreamName('right')
    xout_depth       .setStreamName('depth')
    xout_disparity   .setStreamName('disparity')
    xout_rectif_left .setStreamName('rectified_left')
    xout_rectif_right.setStreamName('rectified_right')
    cam_left .out        .link(stereo.left)
    cam_right.out        .link(stereo.right)
    stereo.syncedLeft    .link(xout_left.input)
    stereo.syncedRight   .link(xout_right.input)
    stereo.depth         .link(xout_depth.input)
    stereo.disparity     .link(xout_disparity.input)
    stereo.rectifiedLeft .link(xout_rectif_left.input)
    stereo.rectifiedRight.link(xout_rectif_right.input)

    streams = ['left', 'right']
    if out_rectified:
        streams.extend(['rectified_left', 'rectified_right'])
    streams.extend(['disparity', 'depth'])

    return pipeline, streams

# The operations done here seem very CPU-intensive, TODO
def convert_to_cv2_frame(name, image,disp_frame_count):
    global last_rectif_right
    baseline = 27 #mm
    focal = right_intrinsic[0][0]
    max_disp = 96
    disp_type = np.uint8
    disp_levels = 1
    if (extended):
        max_disp *= 2
    if (subpixel):
        max_disp *= 32
        disp_type = np.uint16  # 5 bits fractional disparity
        disp_levels = 32

    data, w, h = image.getData(), image.getWidth(), image.getHeight()
    # TODO check image frame type instead of name
    if name == 'rgb_preview':
        frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
    elif name == 'rgb_video': # YUV NV12
        disp_frame_count+=1
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        if disp_frame_count>50:
            cv2.imwrite(str(dest)+'/'+'rgb'+str(disp_frame_count-50)+'.png', frame)
    elif name == 'depth':
        # TODO: this contains FP16 with (lrcheck or extended or subpixel)
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
    elif name == 'disparity':
        disp_frame_count+=1
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))

        # Compute depth from disparity (32 levels)
        with np.errstate(divide='ignore'): # Should be safe to ignore div by zero here
            depth = (disp_levels * baseline * focal / disp).astype(np.uint16)
            #save depth frame as np array
            if disp_frame_count>50:
                np.save(str(dest)+'/'+'depth'+str(disp_frame_count-50)+'.npy', depth)
        if 1: # Optionally, extend disparity range to better visualize it
            frame = (disp * 255. / max_disp).astype(np.uint8)

        if 1: # Optionally, apply a color map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            #frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            #save frame as jpg
            if disp_frame_count>50:
                cv2.imwrite(str(dest)+'/'+'disp_map'+str(disp_frame_count-50)+'.png', frame)
    else: # mono streams / single channel
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        if name.startswith('rectified_'):
            frame = cv2.flip(frame, 1)
            if disp_frame_count>50:
                cv2.imwrite(str(dest)+'/'+'rect_left'+str(disp_frame_count-50)+'.png', frame)
        if name == 'rectified_right':
            last_rectif_right = frame
            if disp_frame_count>50:
                cv2.imwrite(str(dest)+'/'+'rect_right'+str(disp_frame_count-50)+'.png', last_rectif_right)
    return frame,disp_frame_count

def test_pipeline():
    if collecting=='r':
        pipeline, streams = create_rgb_cam_pipeline()
   #pipeline, streams = create_mono_cam_pipeline()
    elif collecting=='d':
        pipeline, streams = create_stereo_depth_pipeline(source_camera)
    if use_calibration:
        exp_time_mono=calib_mono['exp_time']
        sens_iso_mono=calib_mono['sens_iso']
        exp_time_color=calib_color['exp_time']
        sens_iso_color=calib_color['sens_iso']
        lens_pos_color=calib_color['lens_pos']
    print("Creating DepthAI device")
    with dai.Device(pipeline) as device:
        print("Starting pipeline")
        device.startPipeline()
        if collecting=='d' and use_calibration:
            controlQueue_m = device.getInputQueue('control_m')
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            #ctrl.setManualExposure(exp_time_mono, sens_iso_mono)
            controlQueue_m.send(ctrl)
        elif collecting=='r' and use_calibration:
            controlQueue_r = device.getInputQueue('control_r')
            ctrl1 = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            #ctrl1.setManualExposure(exp_time_color, sens_iso_color)
            if focus_mode==0:
                ctrl1.setManualFocus(lens_pos_color)
            else:
                ctrl1.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode(int(focus_mode)))
            controlQueue_r.send(ctrl1)
        # Create a receive queue for each stream
        q_list = []
        for s in streams:
            q = device.getOutputQueue(s, 8, blocking=False)
            q_list.append(q)

        # Need to set a timestamp for input frames, for the sync stage in Stereo node
        disp_frame_count=0
        while True:
            # Handle output streams
            for q in q_list:
                name  = q.getName()
                image = q.get()
                #print("Received frame:", name)
                # Skip some streams for now, to reduce CPU load
                if name in ['left', 'right', 'depth']: continue
                frame,disp_frame_count = convert_to_cv2_frame(name, image,disp_frame_count)
                #cv2.imshow(name, frame)
                if disp_frame_count>=(int(n)+50):
                    break
            if disp_frame_count>=(int(n)+50):
                break
            if cv2.waitKey(1) == ord('q'):
                break
test_pipeline()
