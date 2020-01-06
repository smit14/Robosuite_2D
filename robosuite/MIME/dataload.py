import numpy as np
import json

root_path = '/home/smit/Desktop/CLVR/Compositional_Imitation/MIME/'
path_joint_velocity = root_path + 'joint_angles.txt'
path_right_gripper = root_path + 'right_gripper.txt'
path_left_gripper = root_path + 'right_gripper.txt'

f_joint_velocity = open(path_joint_velocity, 'r')
f_right_gripper = open(path_right_gripper,'r')
f_left_gripper = open(path_left_gripper,'r')

joint_velocity_raw = f_joint_velocity.readlines()
right_gripper_values=  f_right_gripper.readlines()
left_gripper_values = f_left_gripper.readlines()

# n = min(len(joint_velocity_raw), len(right_gripper_values), len(left_gripper_values))
#
# joint_velocity_raw = joint_velocity_raw[:n]
# right_gripper_values = right_gripper_values[:n]
# left_gripper_values = left_gripper_values[:n]


def convert_raw_to_dict(joint_velocity_raw):
    joint_velocity_values = []
    for i in range(n):
        s = joint_velocity_raw[i][:-1]
        joint_velocity_values.append(json.loads(s))

    return joint_velocity_values

joint_velocity_values = convert_raw_to_dict(joint_velocity_raw)

