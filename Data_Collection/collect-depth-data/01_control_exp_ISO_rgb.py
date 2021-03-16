#!/usr/bin/env python3

# This example shows usage of Camera Control message as well as ColorCamera configInput to change crop x and y
# Uses 'WASD' controls to move the crop window and 'C' to capture a still image
import depthai as dai
import cv2
import numpy as np
from tempfile import TemporaryFile


# Step size ('W','A','S','D' controls)
STEP_SIZE = 8
# Manual exposure/ focus step
EXP_STEP = 500 #us
ISO_STEP = 50
LENS_STEP = 3


pipeline = dai.Pipeline()

# Nodes
colorCam = pipeline.createColorCamera()
controlIn = pipeline.createXLinkIn()
configIn = pipeline.createXLinkIn()
videoEncoder = pipeline.createVideoEncoder()
stillEncoder = pipeline.createVideoEncoder()
videoMjpegOut = pipeline.createXLinkOut()
stillMjpegOut = pipeline.createXLinkOut()
previewOut = pipeline.createXLinkOut()


# Properties
colorCam.setVideoSize(640, 360)
colorCam.setPreviewSize(300, 300)
controlIn.setStreamName('control')
configIn.setStreamName('config')
videoEncoder.setDefaultProfilePreset(colorCam.getVideoSize(), colorCam.getFps(), dai.VideoEncoderProperties.Profile.MJPEG)
stillEncoder.setDefaultProfilePreset(colorCam.getStillSize(), 1, dai.VideoEncoderProperties.Profile.MJPEG)
videoMjpegOut.setStreamName('video')
stillMjpegOut.setStreamName('still')
previewOut.setStreamName('preview')


# Link nodes
colorCam.video.link(videoEncoder.input)
colorCam.still.link(stillEncoder.input)
colorCam.preview.link(previewOut.input)
controlIn.out.link(colorCam.inputControl)
configIn.out.link(colorCam.inputConfig)
videoEncoder.bitstream.link(videoMjpegOut.input)
stillEncoder.bitstream.link(stillMjpegOut.input)

def clamp(num, v0, v1): return max(v0, min(num, v1))

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as dev:

    # Get data queues
    controlQueue = dev.getInputQueue('control')
    configQueue = dev.getInputQueue('config')
    previewQueue = dev.getOutputQueue('preview')
    videoQueue = dev.getOutputQueue('video')
    stillQueue = dev.getOutputQueue('still')

    # Start pipeline
    dev.startPipeline()

    # Max crop_x & crop_y
    max_crop_x = (colorCam.getResolutionWidth() - colorCam.getVideoWidth()) / colorCam.getResolutionWidth()
    max_crop_y = (colorCam.getResolutionHeight() - colorCam.getVideoHeight()) / colorCam.getResolutionHeight()

    # Default crop
    crop_x = 0
    crop_y = 0

    default_wb=0
    
    ae_comp = 0
    lens_pos = 150
    lens_min = 0
    lens_max = 255

    exp_time = 20000
    exp_min = 1
    exp_max = 33000

    sens_iso = 800
    sens_min = 100
    sens_max = 1600

    ae_lock = False
    awb_lock = False

    while True:

        previewFrames = previewQueue.tryGetAll()
        for previewFrame in previewFrames:
            cv2.imshow('preview', previewFrame.getData().reshape(previewFrame.getWidth(), previewFrame.getHeight(), 3))

        videoFrames = videoQueue.tryGetAll()
        for videoFrame in videoFrames:
            # Decode JPEG
            frame = cv2.imdecode(videoFrame.getData(), cv2.IMREAD_UNCHANGED)
            # Display
            cv2.imshow('video', frame)

            # Send new cfg to camera
            cfg = dai.ImageManipConfig()
            configQueue.send(cfg)
        stillFrames = stillQueue.tryGetAll()
        for stillFrame in stillFrames:
            # Decode JPEG
            frame = cv2.imdecode(stillFrame.getData(), cv2.IMREAD_UNCHANGED)
            # Display
            cv2.imshow('still', frame)


        # Update screen
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
        	#capture still frame
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            controlQueue.send(ctrl)
        elif key == ord('t'):
            print("Autofocus trigger (and disable continuous)")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            controlQueue.send(ctrl)
        elif key == ord('d'):
            print("Autofocus EDOF mode")
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.EDOF)
            controlQueue.send(ctrl)
        elif key in [ord(','), ord('.')]:
            if key == ord(','): lens_pos -= LENS_STEP
            if key == ord('.'): lens_pos += LENS_STEP
            lens_pos = clamp(lens_pos, lens_min, lens_max)
            print("Setting manual focus, lens position:", lens_pos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lens_pos)
            controlQueue.send(ctrl)
        elif key == ord('w'):
        	#circle between white balances
            if awb_lock == False:
                ctrl=dai.CameraControl()
                awbModes=['OFF', 'AUTO', 'INCANDESCENT', 'FLUORESCENT', 'WARM_FLUORESCENT', 'DAYLIGHT', 'CLOUDY_DAYLIGHT', 'TWILIGHT', 'SHADE']
                default_wb+=1
                default_wb%=9 	
                print('White Balance: ',awbModes[default_wb], default_wb)
                print(type(dai.CameraControl.AutoWhiteBalanceMode.INCANDESCENT))
                ctrl.setAutoWhiteBalanceMode(dai.RawCameraControl.AutoWhiteBalanceMode(default_wb))
                controlQueue.send(ctrl)
            else: 	
                print('AutoWhite balance locked')	
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            #set exposure compensation between -9,9 in stops of 1
            if ae_lock == False:
                if key == ord('i'): exp_time -= EXP_STEP
                if key == ord('o'): exp_time += EXP_STEP
                if key == ord('k'): sens_iso -= ISO_STEP
                if key == ord('l'): sens_iso += ISO_STEP
                exp_time = clamp(exp_time, exp_min, exp_max)
                sens_iso = clamp(sens_iso, sens_min, sens_max)
                print("Setting manual exposure, time:", exp_time, "iso:", sens_iso)
                ctrl = dai.CameraControl()
                ctrl.setManualExposure(exp_time, sens_iso)
                controlQueue.send(ctrl)
            else:
                print('exposure Locked')
        elif key == ord('e'):
        	#ae_lock
            ae_lock = not ae_lock
            print('Exposure lock: ', ae_lock)
        elif key == ord('b'):
        	#awb_lock
            awb_lock = not awb_lock
            print('AWB lock: ', awb_lock)
        elif key == ord('s'):
            #save EV and WB values
            #tmp = TemporaryFile(mode="a+b")
            print('Calibration file "RGB_calib.npz" was saved')
            np.savez('RGB_calib', lens_pos=lens_pos,\
                exp_time=exp_time,sens_iso=sens_iso, \
                awbMode=dai.RawCameraControl.AutoWhiteBalanceMode(default_wb),\
                afMode=dai.CameraControl.AutoFocusMode.EDOF)