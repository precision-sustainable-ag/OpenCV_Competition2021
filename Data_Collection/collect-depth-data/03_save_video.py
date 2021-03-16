#!/usr/bin/env python3
import depthai as dai
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-sp", "--savepath", default='/home/sardesaim/collected_depth/', help="output path for depth data")
parser.add_argument("-c","--usecalibration", default=True, help="Use calibrated parameters for 3A for both cameras")
parser.add_argument("-f","--focusMode", default=3, help="Focus Modes. Enter exactly from [0:'MANUAL',1:'AUTO',3:'CONTINUOUS_VIDEO',4:'CONTINUOUS_PICTURE',5:'EDOF']")
parser.add_argument("-mc", "--calibrationfilemono", default='/home/sardesaim/OpenCV_Competition2021/Data_Collection/collect-depth-data/mono_calib.npz')
parser.add_argument("-rc", "--calibrationfilergb", default='/home/sardesaim/OpenCV_Competition2021/Data_Collection/collect-depth-data/RGB_calib.npz')
args = parser.parse_args()

sp=args.savepath

use_calibration=args.usecalibration
if use_calibration:
    calib_mono=np.load(args.calibrationfilemono)
    calib_color=np.load(args.calibrationfilergb)
focus_mode=args.focusMode

pipeline = dai.Pipeline()

# Nodes
control_inr = pipeline.createXLinkIn()
control_inr.setStreamName('control_r')
control_inm = pipeline.createXLinkIn()
control_inm.setStreamName('control_m')
colorCam = pipeline.createColorCamera()
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
control_inr.out.link(colorCam.inputControl)

monoCam = pipeline.createMonoCamera()
monoCam2 = pipeline.createMonoCamera()
control_inm.out.link(monoCam.inputControl)
control_inm.out.link(monoCam2.inputControl)

ve1 = pipeline.createVideoEncoder()
ve2 = pipeline.createVideoEncoder()
ve3 = pipeline.createVideoEncoder()

ve1Out = pipeline.createXLinkOut()
ve2Out = pipeline.createXLinkOut()
ve3Out = pipeline.createXLinkOut()

# Properties
monoCam.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoCam2.setBoardSocket(dai.CameraBoardSocket.RIGHT)
ve1Out.setStreamName('ve1Out')
ve2Out.setStreamName('ve2Out')
ve3Out.setStreamName('ve3Out')

#setting to 26fps will trigger error
ve1.setDefaultProfilePreset(1280, 720, 25, dai.VideoEncoderProperties.Profile.H264_MAIN)
ve2.setDefaultProfilePreset(3840, 2160, 25, dai.VideoEncoderProperties.Profile.H265_MAIN)
ve3.setDefaultProfilePreset(1280, 720, 25, dai.VideoEncoderProperties.Profile.H264_MAIN)

# Link nodes
monoCam.out.link(ve1.input)
colorCam.video.link(ve2.input)
monoCam2.out.link(ve3.input)

ve1.bitstream.link(ve1Out.input)
ve2.bitstream.link(ve2Out.input)
ve3.bitstream.link(ve3Out.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as dev:

    # Prepare data queues
    outQ1 = dev.getOutputQueue('ve1Out', maxSize=30, blocking=True)
    outQ2 = dev.getOutputQueue('ve2Out', maxSize=30, blocking=True)
    outQ3 = dev.getOutputQueue('ve3Out', maxSize=30, blocking=True)

    if use_calibration:
        exp_time_mono=calib_mono['exp_time']
        sens_iso_mono=calib_mono['sens_iso']
        exp_time_mono=calib_mono['exp_time']
        sens_iso_mono=calib_mono['sens_iso']
        exp_time_color=calib_color['exp_time']
        sens_iso_color=calib_color['sens_iso']
        lens_pos_color=calib_color['lens_pos']
        controlQueue_m = dev.getInputQueue('control_m')
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(exp_time_mono, sens_iso_mono)
        controlQueue_m.send(ctrl)
        controlQueue_r = dev.getInputQueue('control_r')
        ctrl1 = dai.CameraControl()
        ctrl1.setManualExposure(exp_time_color, sens_iso_color)
        if focus_mode==0:
            ctrl1.setManualFocus(lens_pos_color)
        else:
            ctrl1.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode(focus_mode))
        controlQueue_r.send(ctrl1)
    # Start the pipeline
    dev.startPipeline()
    
    # Processing loop
    with open(sp+'mono1.h264', 'wb') as file_mono1_h264, open(sp+'color.h265', 'wb') as file_color_h265, open(sp+'mono2.h264', 'wb') as file_mono2_h264:
        print("Press Ctrl+C to stop encoding...")
        while True:
            try:
                # Empty each queue
                while outQ1.has():
                    outQ1.get().getData().tofile(file_mono1_h264)

                while outQ2.has():
                    outQ2.get().getData().tofile(file_color_h265)

                while outQ3.has():
                    outQ3.get().getData().tofile(file_mono2_h264)
            except KeyboardInterrupt:
                break

    print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
    cmd = "ffmpeg -framerate 25 -i {} -c copy {}"
    print(cmd.format(sp+"mono1.h264", sp+"mono1.mp4"))
    print(cmd.format(sp+"mono2.h264", sp+"mono2.mp4"))
    print(cmd.format(sp+"color.h265", sp+"color.mp4"))