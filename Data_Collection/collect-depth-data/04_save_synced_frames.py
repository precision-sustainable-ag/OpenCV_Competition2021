#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
from time import monotonic
from uuid import uuid4
from multiprocessing import Process, Queue
import cv2
import depthai as dai


def check_range(min_val, max_val):
    def check_fn(value):
        ivalue = int(value)
        if min_val <= ivalue <= max_val:
            return ivalue
        else:
            raise argparse.ArgumentTypeError(
                "{} is an invalid int value, must be in range {}..{}".format(value, min_val, max_val)
            )
    return check_fn


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', default=0.3, type=float, help="Maximum difference between packet timestamps to be considered as synced")
parser.add_argument("-sp", "--savepath", default='/home/pi/collected_depth/', help="output path for depth data")
# parser.add_argument("-n", "--numframes", default=5, help="Number of frames to be saved")
parser.add_argument('-d', '--dirty', action='store_true', default=False, help="Allow the destination path not to be empty")
parser.add_argument('-nd', '--no-debug', dest="prod", action='store_true', default=False, help="Do not display debug output")
parser.add_argument('-m', '--time', type=float, default=float("inf"), help="Finish execution after X seconds")
parser.add_argument('-af', '--autofocus', type=str, default=None, help="Set AutoFocus mode of the RGB camera", choices=list(filter(lambda name: name[0].isupper(), vars(dai.CameraControl.AutoFocusMode))))
# parser.add_argument('-mf', '--manualfocus', type=check_range(0, 255), help="Set manual focus of the RGB camera [0..255]")
# parser.add_argument('-et', '--exposure-time', type=check_range(1, 33000), help="Set manual exposure time of the RGB camera [1..33000]")
# parser.add_argument('-ei', '--exposure-iso', type=check_range(100, 1600), help="Set manual exposure ISO of the RGB camera [100..1600]")
parser.add_argument("-c","--usecalibration", default=False, help="Use calibrated parameters for 3A for both cameras")
parser.add_argument("-mc", "--calibrationfilemono", default='/home/pi/mono_calib.npz')
parser.add_argument("-rc", "--calibrationfilergb", default='/home/pi/RGB_calib.npz')
args = parser.parse_args()

focus_mode=args.autofocus

use_calibration=args.usecalibration
if use_calibration:
    calib_mono=np.load(args.calibrationfilemono)
    calib_color=np.load(args.calibrationfilergb)
    exp_time_mono=calib_mono['exp_time']
    sens_iso_mono=calib_mono['sens_iso']
    exposure_mono=[exp_time_mono, sens_iso_mono]
    exp_time_color=calib_color['exp_time']
    sens_iso_color=calib_color['sens_iso']
    exposure_color=[exp_time_color,sens_iso_color]
    manual_focus=calib_color['lens_pos']

# Depth parameters
out_rectified  = True   # Output and display rectified streams
lrcheck  = True   # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True   # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median   = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# Camera parameters 
right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]
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


dest = Path(args.savepath).resolve().absolute()
dest_count = len(list(dest.glob('*')))
if dest.exists() and dest_count != 0 and not args.dirty:
    raise ValueError(f"Path {dest} contains {dest_count} files. Either specify new path or use \"--dirty\" flag to use current one")
dest.mkdir(parents=True, exist_ok=True)

pipeline = dai.Pipeline()

rgb = pipeline.createColorCamera()
rgb.setPreviewSize(1920, 1080)
rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb.setInterleaved(False)
rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

controlInRgb = pipeline.createXLinkIn()
controlInRgb.setStreamName('control_rgb')
controlInRgb.out.link(rgb.inputControl)
controlInMono = pipeline.createXLinkIn()
controlInMono.setStreamName('control_mono')
controlInMono.out.link(left.inputControl)
controlInMono.out.link(right.inputControl)

depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(255)
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
depth.setMedianFilter(median)
depth.setLeftRightCheck(lrcheck)
depth.setExtendedDisparity(extended)
depth.setSubpixel(subpixel)

left.out.link(depth.left)
right.out.link(depth.right)
# Create output
rgbOut = pipeline.createXLinkOut()
rgbOut.setStreamName("color")
rgb.preview.link(rgbOut.input)
leftOut = pipeline.createXLinkOut()
leftOut.setStreamName("left")
left.out.link(leftOut.input)
rightOut = pipeline.createXLinkOut()
rightOut.setStreamName("right")
right.out.link(rightOut.input)
depthOut = pipeline.createXLinkOut()
depthOut.setStreamName("disparity")
depth.disparity.link(depthOut.input)

# https://stackoverflow.com/a/7859208/5494277
def step_norm(value):
    return round(value / args.threshold) * args.threshold


def seq(packet):
    return packet.getSequenceNum()


def tst(packet):
    return packet.getTimestamp().total_seconds()


# https://stackoverflow.com/a/10995203/5494277
def has_keys(obj, keys):
    return all(stream in obj for stream in keys)


class PairingSystem:
    seq_streams = ["left", "right", "disparity"]
    ts_streams = ["color"]
    seq_ts_mapping_stream = "left"

    def __init__(self):
        self.ts_packets = {}
        self.seq_packets = {}
        self.last_paired_ts = None
        self.last_paired_seq = None

    def add_packets(self, packets, stream_name):
        if packets is None:
            return
        if stream_name in self.seq_streams:
            for packet in packets:
                seq_key = seq(packet)
                self.seq_packets[seq_key] = {
                    **self.seq_packets.get(seq_key, {}),
                    stream_name: packet
                }
        elif stream_name in self.ts_streams:
            for packet in packets:
                ts_key = step_norm(tst(packet))
                self.ts_packets[ts_key] = {
                    **self.ts_packets.get(ts_key, {}),
                    stream_name: packet
                }

    def get_pairs(self):
        results = []
        for key in list(self.seq_packets.keys()):
            if has_keys(self.seq_packets[key], self.seq_streams):
                ts_key = step_norm(tst(self.seq_packets[key][self.seq_ts_mapping_stream]))
                if ts_key in self.ts_packets and has_keys(self.ts_packets[ts_key], self.ts_streams):
                    results.append({
                        **self.seq_packets[key],
                        **self.ts_packets[ts_key]
                    })
                    self.last_paired_seq = key
                    self.last_paired_ts = ts_key
        if len(results) > 0:
            self.collect_garbage()
        return results

    def collect_garbage(self):
        for key in list(self.seq_packets.keys()):
            if key <= self.last_paired_seq:
                del self.seq_packets[key]
        for key in list(self.ts_packets.keys()):
            if key <= self.last_paired_ts:
                del self.ts_packets[key]


extract_frame = {
    "left": lambda item: item.getCvFrame(),
    "right": lambda item: item.getCvFrame(),
    "color": lambda item: item.getCvFrame(),
    "disparity": lambda item: item.getFrame(),
}

frame_q = Queue()


def store_frames(in_q):
    
    while True:
        frames_dict = in_q.get()
        if frames_dict is None:
            return
        frames_path = dest / Path(str(uuid4()))
        frames_path.mkdir(parents=False, exist_ok=False)
        for stream_name, item in frames_dict.items():
            if stream_name=="disparity":
                disp=item.astype(uint8)
                depth=(disp_levels * baseline * focal / disp).astype(np.uint16)
                np.save(str(frames_path / Path(f"{stream_name}.npy")), depth)
                dispmap=cv2.applyColorMap(item, cv2.COLORMAP_JET)
                cv2.imwrite(str(frames_path / Path(f"{stream_name}.png")), dispmap)
            else:
                cv2.imwrite(str(frames_path / Path(f"{stream_name}.png")), item)

store_p = Process(target=store_frames, args=(frame_q, ))
store_p.start()
ps = PairingSystem()

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()
    if use_calibration:
        qControlRGB = device.getInputQueue('control_rgb')
        ctrl = dai.CameraControl()
        if args.autofocus:
            ctrl.setAutoFocusMode(getattr(dai.CameraControl.AutoFocusMode, args.autofocus))
        if all(exposure_color):
            ctrl.setManualExposure(*exposure_color)
            ctrl.setManualFocus(manual_focus)
        qControlRGB.send(ctrl)

        qControlMono = device.getInputQueue('control_mono')
        ctrl1 = dai.CameraControl()
        if all(exposure_mono):
            ctrl1.setManualExposure(*exposure_mono)
        qControlMono.send(ctrl1)

    cfg = dai.ImageManipConfig()
    ctrl = dai.CameraControl()

    start_ts = monotonic()
    while True:
        for queueName in PairingSystem.seq_streams + PairingSystem.ts_streams:
            ps.add_packets(device.getOutputQueue(queueName).tryGetAll(), queueName)

        pairs = ps.get_pairs()
        for pair in pairs:
            extracted_pair = {stream_name: extract_frame[stream_name](item) for stream_name, item in pair.items()}
            if not args.prod:
                for stream_name, item in extracted_pair.items():
                    cv2.imshow(stream_name, item)
            frame_q.put(extracted_pair)

        if not args.prod and cv2.waitKey(1) == ord('q'):
            break

        if monotonic() - start_ts > args.time:
            break

frame_q.put(None)
store_p.join()
