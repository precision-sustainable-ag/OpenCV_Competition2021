import sys

sys.path.append("..")
from MachineMotion import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--distance', type=float, help='Relative distance that the MM has to move')
parser.add_argument('-hm', '--home', help='Indicates the MM to go to home', action='store_true')
args = parser.parse_args()

###################### Machine Motion initialization ######################

# Initialize the machine motion object
mm = MachineMotion(DEFAULT_IP)
print('mm initialized')

# Remove the software stop
print("--> Removing software stop")
mm.releaseEstop()
print("--> Resetting system")
mm.resetSystem()

# Configure the axis number 1, 8 uSteps and 10 mm / turn for a ball screw linear drive
axis = AXIS_NUMBER.DRIVE2
uStep = MICRO_STEPS.ustep_8
mechGain = MECH_GAIN.ballscrew_10mm_turn
mm.configAxis(axis, uStep, mechGain)
print("Axis " + str(axis) + " configured with " + str(uStep) + " microstepping and " + str(
    mechGain) + "mm/turn mechanical gain")

# Configure the axis direction
ax_direction = DIRECTION.POSITIVE
mm.configAxisDirection(axis, ax_direction)
print("Axis direction set to " + str(ax_direction))

# Relative Move Parameters
speed = 10
acceleration = 100

# Load Relative Move Parameters
mm.emitSpeed(speed)
mm.emitAcceleration(acceleration)

###################### End of Machine Motion initialization ######################

if args.distance:
    # Start the relative move
    distance = args.distance
    direction = 'positive'
    mm.emitRelativeMove(axis, direction, distance)
    mm.waitForMotionCompletion()
    mm.emitStop()
    distance_home = mm.getCurrentPositions()[2]
    print(f'You are {distance_home} mm from home')

if args.home:
    # Start movement in negative direction
    if mm.getCurrentPositions()[2] != 0.0:
        distance_home = mm.getCurrentPositions()[2]
        mm.emitRelativeMove(axis, 'negative', distance_home)
        mm.waitForMotionCompletion()
        mm.emitStop()
        current_position = mm.getCurrentPositions()[2]
        print(f'mm is at{current_position}')
    else:
        print('mm is at home')

    if mm.getCurrentPositions()[2] != 0.0:
        mm.configHomingSpeed(axis, 5)
        mm.emitHome(axis)
        print('mm is going home')
        mm.waitForMotionCompletion()
        print('mm is at home')
        mm.emitStop()
    else:
        print('mm is already at home')

    mm.triggerEstop()