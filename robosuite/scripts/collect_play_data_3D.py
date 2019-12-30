"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import os
import shutil
import time
import argparse
import datetime
import h5py
from glob import glob
import numpy as np

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import IKWrapper
from robosuite.wrappers import DataCollectionWrapper


def cube_outside(cur_obs, lower_bound, upper_bound, cube_init_z_pos):
	cubeA_pos_x = cur_obs["cubeA_pos"][0]
	cubeB_pos_x = cur_obs["cubeB_pos"][0]
	cubeC_pos_x = cur_obs["cubeC_pos"][0]
	cubeD_pos_x = cur_obs["cubeD_pos"][0]
	cubeE_pos_x = cur_obs["cubeE_pos"][0]

	cubeA_pos_z = cur_obs["cubeA_pos"][2]
	cubeB_pos_z = cur_obs["cubeB_pos"][2]
	cubeC_pos_z = cur_obs["cubeC_pos"][2]
	cubeD_pos_z = cur_obs["cubeD_pos"][2]
	cubeE_pos_z = cur_obs["cubeE_pos"][2]

	cube_list = []
	if cubeA_pos_z==cube_init_z_pos and (cubeA_pos_x < lower_bound or cubeA_pos_x > upper_bound):
		cube_list.append("cubeA")
	if cubeB_pos_z==cube_init_z_pos and (cubeB_pos_x < lower_bound or cubeB_pos_x > upper_bound):
		cube_list.append("cubeB")
	if cubeC_pos_z==cube_init_z_pos and (cubeC_pos_x < lower_bound or cubeC_pos_x > upper_bound):
		cube_list.append("cubeC")
	if cubeD_pos_z==cube_init_z_pos and (cubeD_pos_x < lower_bound or cubeD_pos_x > upper_bound):
		cube_list.append("cubeD")
	if cubeE_pos_z==cube_init_z_pos and (cubeE_pos_x < lower_bound or cubeE_pos_x > upper_bound):
		cube_list.append("cubeE")
	return cube_list


def cube_tilted(cur_obs, cube_init_z_pos):
	cubeA_rot = cur_obs["cubeA_quat"]
	cubeB_rot = cur_obs["cubeB_quat"]
	cubeC_rot = cur_obs["cubeC_quat"]
	cubeD_rot = cur_obs["cubeD_quat"]
	cubeE_rot = cur_obs["cubeE_quat"]

	cubeA_pos_z = cur_obs["cubeA_pos"][2]
	cubeB_pos_z = cur_obs["cubeB_pos"][2]
	cubeC_pos_z = cur_obs["cubeC_pos"][2]
	cubeD_pos_z = cur_obs["cubeD_pos"][2]
	cubeE_pos_z = cur_obs["cubeE_pos"][2]

	ideal_rot = np.array([0,0,0,1.0])

	cube_list = []
	if(np.sum(cubeA_rot-ideal_rot)>0 and cubeA_pos_z==cube_init_z_pos):
		cube_list.append("cubeA")
	if(np.sum(cubeB_rot-ideal_rot)>0 and cubeB_pos_z==cube_init_z_pos):
		cube_list.append("cubeB")
	if (np.sum(cubeC_rot - ideal_rot) > 0 and cubeC_pos_z==cube_init_z_pos):
		cube_list.append("cubeC")
	if (np.sum(cubeD_rot - ideal_rot) > 0 and cubeD_pos_z==cube_init_z_pos):
		cube_list.append("cubeD")
	if (np.sum(cubeE_rot - ideal_rot) > 0 and cubeE_pos_z==cube_init_z_pos):
		cube_list.append("cubeE")
	return cube_list

def collect_human_trajectory(env, device):
	"""
	Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
	The rollout trajectory is saved to files in npz format.
	Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

	Args:
		env: environment to control
		device (instance of Device class): to receive controls from the device
	"""

	obs = env.reset()
	camera_id = 0

	# Arm initial position
	# for x-y-z axis
	# init_join_position = [0, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708]

	# for y-z axis
	init_joint_position = [-0.59382754, -1.12190546,  0.48425191,  1.99674156, -0.2968217,   0.76457908,  1.82085369]
	env.set_robot_joint_positions(init_joint_position)

	env.viewer.set_camera(camera_id=camera_id)
	env.render()

	is_first = True

	# never terminate episode
	reset = False
	device.start_control()

	# initial z position of cubes
	cube_init_z_pos = obs["cubeA_pos"][2]

	#lower and upper bound on x-axis for cube position
	lower_bound_x = 0.545
	upper_bound_x = 0.575

	# cool down after changing dpos[0]
	cool_down_period = 10
	cool_down_counter = 0

	while not reset:
		state = device.get_controller_state()


		ideal_arm_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
		cur_arm_pos = env._right_hand_pos
		cur_arm_pos_x = cur_arm_pos[0]

		if cool_down_counter>0:
			cool_down_counter-=1

		# convert first dimension(x-axis) to appropriate values (if arm is too far from plane then apply reverse dpos action)
		if cur_arm_pos_x<lower_bound_x and cool_down_counter==0:
			state["dpos"][0] = 0.01
			cool_down_counter = 10
		elif cur_arm_pos_x>upper_bound_x and cool_down_counter==0:
			state["dpos"][0] = -0.01
			cool_down_counter = 10
		else:
			state["dpos"][0] = 0

		# disable the rotation of the end effector
		state["rotation"] = ideal_arm_rotation

		# ideal cube position : [0.56, yy, zz]

		# check if any cubes are out of the plane
		cur_obs = env.unwrapped._get_observation()

		outside_cube_name = cube_outside(cur_obs, lower_bound_x, upper_bound_x, cube_init_z_pos)
		tilted_cube_name = cube_tilted(cur_obs,cube_init_z_pos)
		# print only if change in position
		# if(np.sum(state["dpos"]!=0)>0):
		# 	print(state["dpos"])
		# 	print(state["grasp"])
		# 	print()

		dpos, rotation, grasp, reset = (
			state["dpos"],
			state["rotation"],
			state["grasp"],
			state["reset"],
		)

		# convert into a suitable end effector action for the environment
		current = env._right_hand_orn
		drotation = current.T.dot(rotation)  # relative rotation of desired from current
		dquat = T.mat2quat(drotation)
		grasp = grasp - 1.  # map 0 to -1 (open) and 1 to 0 (closed halfway)
		action = np.concatenate([dpos, dquat, [grasp]])

		obs, reward, done, info = env.step(action)

		if is_first:
			is_first = False

			# We grab the initial model xml and state and reload from those so that
			# we can support deterministic playback of actions from our demonstrations.
			# This is necessary due to rounding issues with the model xml and with
			# env.sim.forward(). We also have to do this after the first action is
			# applied because the data collector wrapper only starts recording
			# after the first action has been played.
			initial_mjstate = env.sim.get_state().flatten()
			xml_str = env.model.get_xml()
			env.reset_from_xml_string(xml_str)
			env.sim.reset()
			env.sim.set_state_from_flattened(initial_mjstate)
			env.sim.forward()
			env.viewer.set_camera(camera_id=camera_id)

			# set the conaffinity and contype of walls and objects
			id2name_dict = env.unwrapped.sim.model._geom_id2name
			for i in range(51):
				if id2name_dict[i] is None or i >= 49:		# if its arm's body part or wall
					env.unwrapped.sim.model.geom_conaffinity[i] = 1
					env.unwrapped.sim.model.geom_contype[i] = 0

			env.unwrapped.sim.model.geom_group[49:51] = 0	# turn the wall invisible

		env.render()
	# cleanup for end of data collection episodes
	env.close()


def gather_demonstrations_as_hdf5(directory, out_dir):
	"""
	Gathers the demonstrations saved in @directory into a
	single hdf5 file, and another directory that contains the
	raw model.xml files.

	The strucure of the hdf5 file is as follows.

	data (group)
		date (attribute) - date of collection
		time (attribute) - time of collection
		repository_version (attribute) - repository version used during collection
		env (attribute) - environment name on which demos were collected

		demo1 (group) - every demonstration has a group
			model_file (attribute) - name of corresponding model xml in `models` directory
			states (dataset) - flattened mujoco states
			joint_velocities (dataset) - joint velocities applied during demonstration
			gripper_actuations (dataset) - gripper controls applied during demonstration
			right_dpos (dataset) - end effector delta position command for
				single arm robot or right arm
			right_dquat (dataset) - end effector delta rotation command for
				single arm robot or right arm
			left_dpos (dataset) - end effector delta position command for
				left arm (bimanual robot only)
			left_dquat (dataset) - end effector delta rotation command for
				left arm (bimanual robot only)

		demo2 (group)
		...

	Args:
		directory (str): Path to the directory containing raw demonstrations.
		out_dir (str): Path to where to store the hdf5 file and model xmls.
			The model xmls will be stored in a subdirectory called `models`.
	"""

	# store model xmls in this directory
	model_dir = os.path.join(out_dir, "models")
	if os.path.isdir(model_dir):
		shutil.rmtree(model_dir)
	os.makedirs(model_dir)

	hdf5_path = os.path.join(out_dir, "demo.hdf5")
	f = h5py.File(hdf5_path, "w")

	# store traj_per_file
	traj_per_file = [1]
	f.create_dataset("traj_per_file",data=np.array(traj_per_file))

	# store trajectory info in traj0 folder/group
	grp = f.create_group("traj0")

	num_eps = 0
	env_name = None  # will get populated at some point

	for ep_directory in os.listdir(directory):

		state_paths = os.path.join(directory, ep_directory, "state_*.npz")
		states = []
		joint_velocities = []
		gripper_actuations = []
		right_dpos = []
		right_dquat = []
		left_dpos = []
		left_dquat = []

		for state_file in sorted(glob(state_paths)):
			dic = np.load(state_file, allow_pickle=True)
			env_name = str(dic["env"])

			states.extend(dic["states"])
			for ai in dic["action_infos"]:
				joint_velocities.append(ai["joint_velocities"])
				gripper_actuations.append(ai["gripper_actuation"])
				right_dpos.append(ai.get("right_dpos", []))
				right_dquat.append(ai.get("right_dquat", []))
				left_dpos.append(ai.get("left_dpos", []))
				left_dquat.append(ai.get("left_dquat", []))

		if len(states) == 0:
			continue

		# Delete the first actions and the last state. This is because when the DataCollector wrapper
		# recorded the states and actions, the states were recorded AFTER playing that action.
		del states[-1]
		del joint_velocities[0]
		del gripper_actuations[0]
		del right_dpos[0]
		del right_dquat[0]
		del left_dpos[0]
		del left_dquat[0]

		num_eps += 1

		ep_data_grp = grp

		# store model file name as an attribute
		ep_data_grp.attrs["model_file"] = "model_{}.xml".format(num_eps)

		# write datasets for states
		ep_data_grp.create_dataset("states", data=np.array(states))

		# write dataset for joint velocities
		ep_data_grp.create_dataset("joint_velocities", data=np.array(joint_velocities))

		#create dataset of actions: [dpos, dquat, gripper_actuations]

		dpos_np = np.array(right_dpos)
		dquat_np = np.array(right_dquat)
		gripper_actuations_np = np.array(gripper_actuations)

		actions = np.hstack((dpos_np,dquat_np,gripper_actuations_np))
		ep_data_grp.create_dataset("actions",data=actions)

		# write dataset for pad_mask
		niter = actions.shape[0]
		pad_mask = [1]*niter
		ep_data_grp.create_dataset("pad_mask",data=np.array(pad_mask))

		# write dummy dataset for images
		ep_data_grp.create_dataset("images",data=np.array([1]))

		# copy over and rename model xml
		xml_path = os.path.join(directory, ep_directory, "model.xml")
		shutil.copy(xml_path, model_dir)
		os.rename(
			os.path.join(model_dir, "model.xml"),
			os.path.join(model_dir, "model_{}.xml".format(num_eps)),
		)

	# write dataset attributes (metadata)
	now = datetime.datetime.now()
	grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
	grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
	grp.attrs["repository_version"] = robosuite.__version__
	grp.attrs["env"] = env_name

	f.close()

def gather_demonstrations_as_hdf5_original(directory, out_dir):
	"""
	Gathers the demonstrations saved in @directory into a
	single hdf5 file, and another directory that contains the
	raw model.xml files.

	The strucure of the hdf5 file is as follows.

	data (group)
		date (attribute) - date of collection
		time (attribute) - time of collection
		repository_version (attribute) - repository version used during collection
		env (attribute) - environment name on which demos were collected

		demo1 (group) - every demonstration has a group
			model_file (attribute) - name of corresponding model xml in `models` directory
			states (dataset) - flattened mujoco states
			joint_velocities (dataset) - joint velocities applied during demonstration
			gripper_actuations (dataset) - gripper controls applied during demonstration
			right_dpos (dataset) - end effector delta position command for
				single arm robot or right arm
			right_dquat (dataset) - end effector delta rotation command for
				single arm robot or right arm
			left_dpos (dataset) - end effector delta position command for
				left arm (bimanual robot only)
			left_dquat (dataset) - end effector delta rotation command for
				left arm (bimanual robot only)

		demo2 (group)
		...

	Args:
		directory (str): Path to the directory containing raw demonstrations.
		out_dir (str): Path to where to store the hdf5 file and model xmls.
			The model xmls will be stored in a subdirectory called `models`.
	"""

	# store model xmls in this directory
	model_dir = os.path.join(out_dir, "models")
	if os.path.isdir(model_dir):
		shutil.rmtree(model_dir)
	os.makedirs(model_dir)

	hdf5_path = os.path.join(out_dir, "demo.hdf5")
	f = h5py.File(hdf5_path, "w")

	# store some metadata in the attributes of one group
	grp = f.create_group("data")

	num_eps = 0
	env_name = None  # will get populated at some point

	for ep_directory in os.listdir(directory):

		state_paths = os.path.join(directory, ep_directory, "state_*.npz")
		states = []
		joint_velocities = []
		gripper_actuations = []
		right_dpos = []
		right_dquat = []
		left_dpos = []
		left_dquat = []

		for state_file in sorted(glob(state_paths)):
			dic = np.load(state_file, allow_pickle=True)
			env_name = str(dic["env"])

			states.extend(dic["states"])
			for ai in dic["action_infos"]:
				joint_velocities.append(ai["joint_velocities"])
				gripper_actuations.append(ai["gripper_actuation"])
				right_dpos.append(ai.get("right_dpos", []))
				right_dquat.append(ai.get("right_dquat", []))
				left_dpos.append(ai.get("left_dpos", []))
				left_dquat.append(ai.get("left_dquat", []))

		if len(states) == 0:
			continue

		# Delete the first actions and the last state. This is because when the DataCollector wrapper
		# recorded the states and actions, the states were recorded AFTER playing that action.
		del states[-1]
		del joint_velocities[0]
		del gripper_actuations[0]
		del right_dpos[0]
		del right_dquat[0]
		del left_dpos[0]
		del left_dquat[0]

		num_eps += 1
		ep_data_grp = grp.create_group("demo_{}".format(num_eps))

		# store model file name as an attribute
		ep_data_grp.attrs["model_file"] = "model_{}.xml".format(num_eps)

		# write datasets for states and actions
		ep_data_grp.create_dataset("states", data=np.array(states))
		ep_data_grp.create_dataset("joint_velocities", data=np.array(joint_velocities))
		ep_data_grp.create_dataset(
			"gripper_actuations", data=np.array(gripper_actuations)
		)
		ep_data_grp.create_dataset("right_dpos", data=np.array(right_dpos))
		ep_data_grp.create_dataset("right_dquat", data=np.array(right_dquat))
		ep_data_grp.create_dataset("left_dpos", data=np.array(left_dpos))
		ep_data_grp.create_dataset("left_dquat", data=np.array(left_dquat))

		# copy over and rename model xml
		xml_path = os.path.join(directory, ep_directory, "model.xml")
		shutil.copy(xml_path, model_dir)
		os.rename(
			os.path.join(model_dir, "model.xml"),
			os.path.join(model_dir, "model_{}.xml".format(num_eps)),
		)

	# write dataset attributes (metadata)
	now = datetime.datetime.now()
	grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
	grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
	grp.attrs["repository_version"] = robosuite.__version__
	grp.attrs["env"] = env_name

	f.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--directory",
		type=str,
		default=os.path.join(robosuite.models.assets_root, "demonstrations"),
	)
	parser.add_argument("--environment", type=str, default="SawyerLift")
	parser.add_argument("--device", type=str, default="keyboard")
	args = parser.parse_args()

	# create original environment

	env = robosuite.make(
		args.environment,
		ignore_done=True,
		use_camera_obs=False,
		has_renderer=True,
		control_freq=100,
		gripper_visualization=True,

	)

	# enable controlling the end effector directly instead of using joint velocities
	env = IKWrapper(env)

	# wrap the environment with data collection wrapper
	tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
	env = DataCollectionWrapper(env, tmp_directory)

	# initialize device
	if args.device == "keyboard":
		from robosuite.devices import Keyboard

		device = Keyboard()
		env.viewer.add_keypress_callback("any", device.on_press)
		env.viewer.add_keyup_callback("any", device.on_release)
		env.viewer.add_keyrepeat_callback("any", device.on_press)
	elif args.device == "spacemouse":
		from robosuite.devices import SpaceMouse

		device = SpaceMouse()
	else:
		raise Exception(
			"Invalid device choice: choose either 'keyboard' or 'spacemouse'."
		)

	# make a new timestamped directory
	t1, t2 = str(time.time()).split(".")
	new_dir = os.path.join(args.directory, "{}_{}".format(t1, t2))
	os.makedirs(new_dir)

	# store original hdf5
	new_dir_original = new_dir+"_original"
	# collect demonstrations
	collect_human_trajectory(env, device)
	gather_demonstrations_as_hdf5(tmp_directory, new_dir)
	gather_demonstrations_as_hdf5_original(tmp_directory, new_dir_original)
