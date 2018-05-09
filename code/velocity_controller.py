#!/usr/bin/env python
import rospy

# Imports for bag recording
import rosbag
import subprocess
import os
import sys
import signal
import time

from std_msgs.msg import Int16, Float32
from sensor_msgs.msg import Imu, Joy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import numpy as np

'''
A listener that subscribes to the node publishing the current surface.  When the surface changes, check to see if the
new one is already known.  If so, update the command parameter tuple (speed_coefficient, turn_radius_coefficient) to
the previously discovered values for this surface.  If the surface is unknown, gradually increase the speed of the robot
until the vibration threshold is reached.
'''

default_dict = {"max_x_vel" : 2.0, "max_accel" : 0.25}
#dictionary of surfaces that have been discovered
#keys are integers representing a surface
surface_data = {-1 : default_dict.copy()}
#keep track of the previous surface so each iteration can compare to it
current_surface = -1
previous_surface = -1

# bumpiness tracking
previous_bumpiness = 0
current_bumpiness = 0
upcoming_bumpiness = False
surface_transition = False

start_position = 0

accels = []

current_commanded_velocity = 0

#current amount of vertical vibration the robot is experiencing
#updated by the z acceleration recorded by the IMU
z_vibrations = -1
#The maximum amount of vibration the robot is allowed to experience
vibration_threshold = 0.45

previous_stamp = 0
previous_x_velocity = 0
older_x_velocity = 0

speed_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

### PID Variables ####
previous_error = 0.0

kp = 1
ki = 0.001
kd = 0.00001

p_error = 0.0
i_error = 0.0
d_error = 0.0

current_x_vel = 0.0

delta_t = 0.0

previous_odometry_reading_time = 0.0

'''
Callback to run every time a new surface is published to the surface_detection topic
'''
def surface_callback(data):
    global current_surface
    global previous_surface
    global surface_data
    global previous_bumpiness
    global current_bumpiness

    previous_surface = current_surface
    current_surface = data.data

    # if we have been on this surface before, we can update the tuple directly
    if current_surface not in surface_data:
        if previous_surface == -1:
            surface_data[current_surface] = surface_data[-1].copy()
            surface_data[-1] = default_dict.copy()
        else:
	       surface_data[current_surface] = default_dict.copy()
    else:
	   surface_data[-1] = default_dict.copy()

def bumpiness_callback(data):
    global previous_bumpiness
    global current_bumpiness
    global upcoming_bumpiness
    global surface_transition

    previous_bumpiness = current_bumpiness
    current_bumpiness = data.data

    #print(current_bumpiness, previous_bumpiness)
    #print("---------------------")

    # Upcoming bumpier surface
    if current_bumpiness > previous_bumpiness * 1.75 and previous_bumpiness != 0:
        upcoming_bumpiness = True

    # Upcoming less bumpy surface
    if current_bumpiness < previous_bumpiness / 1.75 and previous_bumpiness != 0:
        surface_transition = True

def odometry_callback(data):
    global current_x_vel
    global delta_t
    global previous_odometry_reading_time
    global upcoming_bumpiness
    global start_position
    global surface_transition

    # get the velocity message out of the data
    velocity_msg = data.twist.twist
    # get the x velocity
    current_x_vel = velocity_msg.linear.x
    # calculate the time difference between the previous velocity reading and this one
    current_seconds = data.header.stamp.secs
    current_nanoseconds = data.header.stamp.nsecs
    current_stamp = (10**-9) * current_nanoseconds + current_seconds
    delta_t = current_stamp - previous_odometry_reading_time
    previous_odometry_reading_time = current_stamp

    # distance tracking for upcoming surface
    lidar_look_ahead_distance_m = .45
    current_position = data.pose.pose.position.x

    if (upcoming_bumpiness or surface_transition) and start_position == 0:
        start_position = current_position
        #print("TRANSITION DETECTED")

    if start_position != 0 and abs(current_position - start_position) >= lidar_look_ahead_distance_m:
        upcoming_bumpiness = False
        surface_transition = False
        start_position = 0

def joystick_callback(data):
    global previous_stamp
    global previous_x_velocity
    global older_x_velocity
    global speed_publisher
    global current_surface
    global previous_surface
    global default_dict
    global upcoming_bumpiness

    output_velocity = Twist()

    # If R2 is not pressed down, don't move robot
    if data.buttons[9] == 0:
        return

    input_x_velocity = data.axes[1] * -2
    input_x_direction = 1
    if input_x_velocity < 0:
        input_x_direction = -1
    input_x_velocity = abs(input_x_velocity)

    if upcoming_bumpiness and input_x_velocity > 0.2:
        input_x_velocity = 0.2

    surface_dictionary = surface_data[current_surface]
    output_x_velocity = min(surface_dictionary["max_x_vel"], input_x_velocity) * input_x_direction

    current_seconds = data.header.stamp.secs
    current_nanoseconds = data.header.stamp.nsecs
    current_stamp = (10**-9) * current_nanoseconds + current_seconds
    dt = current_stamp - previous_stamp

    if previous_x_velocity > 2 or previous_x_velocity < -2:
        previous_x_velocity
        sys.quit()

    if dt == 0 or previous_stamp == 0:
        output_velocity.linear.x = previous_x_velocity
        speed_publisher.publish(output_velocity)
        previous_stamp = current_stamp
        return

    previous_stamp = current_stamp
    dv = output_x_velocity - previous_x_velocity
    a = dv/dt

    if abs(a) > surface_dictionary["max_accel"]:
        adjusted_a = surface_dictionary["max_accel"]
        if a < 0:
            adjusted_a *= -1
        output_x_velocity = adjusted_a*dt + previous_x_velocity

    adjusted_velocity = calculate_pid(output_x_velocity)
    output_velocity.linear.x = adjusted_velocity

    older_x_velocity = previous_x_velocity
    previous_x_velocity = adjusted_velocity

    #if (output_velocity.linear.x > 2) or (output_velocity.linear.x < -2):
    #    output_velocity.linear.x = 0
    #    print("ERROR: Speed too high")
    #    sys.exit(1)

    speed_publisher.publish(output_velocity)

def imu_callback(data):
    global accels
    global z_vibrations
    global surface_data
    global current_surface
    global previous_x_velocity
    global older_x_velocity
    global surface_transition

    #print(surface_data)
    z_vibrations = data.linear_acceleration.z
    accels.append(z_vibrations)
    if len(accels) >= 20:
        std_dev = np.std(accels)
        accels = []
        if std_dev > vibration_threshold and abs(older_x_velocity) < abs(previous_x_velocity) and not surface_transition:
            current_max = surface_data[current_surface]["max_x_vel"]
            surface_data[current_surface]["max_x_vel"] = min(abs(previous_x_velocity), current_max) - .15
	    #print("SURFACE " + str(current_surface) + ": UPDATING MAX FROM " + str(current_max) + " to " + str(surface_data[current_surface]["max_x_vel"]))


def calculate_pid(target_velocity):
    global current_x_vel
    global previous_error
    global delta_t
    global p_error
    global i_error
    global d_error
    global kp
    global ki
    global kd

    p_error = target_velocity - current_x_vel
    i_error += p_error*delta_t

    if delta_t > 0:
        d_error = (p_error - previous_error)/delta_t

    previous_error = p_error
    return kp*p_error + ki*i_error + kd*d_error + current_x_vel


'''
Main loop of the listener
'''
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    #print("in listener")
    rospy.init_node('surface_listener', anonymous=True)

    rospy.Subscriber('surface_detection', Int16, surface_callback)
    rospy.Subscriber('surface_bumpiness', Float32, bumpiness_callback)

    #subscribe to the IMU to update the z_vibrations variable
    rospy.Subscriber("/imu/data", Imu, imu_callback)

    #subscribe to the IMU to update the z_vibrations variable
    rospy.Subscriber("/bluetooth_teleop/joy", Joy, joystick_callback)

    rospy.Subscriber("/odometry/filtered", Odometry, odometry_callback)

    # Set up bag recording
    command = "rosbag record /imu/data /odometry/filtered /bluetooth_teleop/joy /feedback /cmd_vel /surface_detection -o final_demo_" + sys.argv[1] +  ".bag"
    p = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True, cwd="./", executable='/bin/bash')

    rospy.sleep(2)

    while not rospy.is_shutdown():
        if (raw_input("Exit? (y/n): ") == 'y'):
            list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
            list_output = list_cmd.stdout.read()
            retcode = list_cmd.wait()
            assert retcode == 0, "List command returned %d" % retcode
            for str in list_output.split("\n"):
                if (str.startswith("/record")):
                    os.system("rosnode kill " + str)
            break


if __name__ == '__main__':
    listener()
