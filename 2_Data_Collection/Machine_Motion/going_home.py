import sys

sys.path.append("..")
from MachineMotion import *


#### Machine Motion initialization ####

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
speed = 50
acceleration = 100

# Load Relative Move Parameters
mm.emitSpeed(speed)
mm.emitAcceleration(acceleration)

if mm.getCurrentPositions()[2] != 0.0:
    distance_home = mm.getCurrentPositions()[2]
    mm.emitRelativeMove(axis, 'negative', distance_home)
    mm.waitForMotionCompletion()
    print('mm is going home')
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