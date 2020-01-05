"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use_actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/SawyerPickPlace/
"""
import os
import h5py
import argparse
import random
import numpy as np
import time

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
import robosuite.utils.transform_utils as T
from robosuite.wrappers import DataCollectionWrapper
from robosuite.wrappers import IKWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default=os.path.join(
            robosuite.models.assets_root, "demonstrations/SawyerNutAssembly"
        ),
    )
    parser.add_argument(
        "--use-actions",
        action='store_true',
    )
    parser.add_argument("--ds",action='store_true', default=False)
    parser.add_argument("--original", action='store_true', default=False)
    parser.add_argument("--ik", action='store_true', default=False)

    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    if args.ds:
        hdf5_path = os.path.join(demo_path, "demo_ds.hdf5")
        print("opening downsampled file")
    f = h5py.File(hdf5_path, "r")
    env_name = f["traj0"].attrs["env"]

    env = robosuite.make(
        env_name,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        reward_shaping=True,
        control_freq=100,
    )

    if args.ik:
        env = IKWrapper(env)

    # lower and upper bound on x-axis for cube position
    lower_bound_x = 0.545
    upper_bound_x = 0.575

    # cool down after changing dpos[0]
    cool_down_period = 10
    cool_down_counter = 0

    while True:
        print("Playing back random episode... (press ESC to quit)")

        # # episode traj0
        ep = "traj0"

        # read the model xml, using the metadata stored in the attribute for this episode
        model_file = f["{}".format(ep)].attrs["model_file"]
        model_path = os.path.join(demo_path, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()

        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)
        env.sim.model.geom_group[49:51] = 0
        # load the flattened mujoco states
        states = f["{}/states".format(ep)].value

        init_joint_position = [-0.59382754, -1.12190546, 0.48425191, 1.99674156, -0.2968217, 0.76457908, 1.82085369]
        env.set_robot_joint_positions(init_joint_position)

        id2name_dict = env.sim.model._geom_id2name
        for i in range(51):
            if id2name_dict[i] is None or i >= 49:  # if its arm's body part or wall
                env.sim.model.geom_conaffinity[i] = 1
                env.sim.model.geom_contype[i] = 0

        env.sim.model.geom_group[49:51] = 0  # turn the wall invisible

        if args.use_actions:

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            # load the actions and play them back open-loop
            jvels = f["{}/joint_velocities".format(ep)].value
            grip_acts = f["{}/actions".format(ep)].value
            grip_acts = grip_acts[:, 7:8]
            actions = np.concatenate([jvels, grip_acts], axis=1)

            if args.ik:
                actions = f["{}/actions".format(ep)].value

            num_actions = actions.shape[0]

            for j, action in enumerate(actions):

                if args.ik:
                    # ideal_arm_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
                    cur_arm_pos = env._right_hand_pos
                    cur_arm_pos_x = cur_arm_pos[0]

                    if cool_down_counter > 0:
                        cool_down_counter -= 1

                    # convert first dimension(x-axis) to appropriate values (if arm is too far from plane then apply reverse dpos action)
                    if cur_arm_pos_x < lower_bound_x and cool_down_counter == 0:
                        action[0] = 0.01
                        cool_down_counter = cool_down_counter
                    elif cur_arm_pos_x > upper_bound_x and cool_down_counter == 0:
                        action[0] = -0.01
                        cool_down_counter = cool_down_counter
                    else:
                        action[0] = 0

                env.step(action)
                env.render()

                if j < num_actions - 1:
                    # ensure that the actions deterministically lead to the same recorded states
                    state_playback = env.sim.get_state().flatten()
                    # assert(np.all(np.equal(states[j + 1], state_playback)))

        else:

            # force the sequence of internal mujoco states one by one
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                env.render()

        break

    f.close()
