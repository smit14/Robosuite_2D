import argparse
import h5py
import random
import os
import datetime
import numpy as np
import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
# import seaborn

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite import make
from robosuite.utils.ffmpeg_gif import save_gif
import robosuite.utils.transform_utils as T


def render(args, f, env):
    demos = list(f["data"].keys())
    for key in tqdm.tqdm(demos):
        # read the model xml, using the metadata stored in the attribute for this episode
        model_file = f["data/{}".format(key)].attrs["model_file"]
        model_path = os.path.join(args.demo_folder, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()

        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

        # load + subsample data
        states, _ = FixedFreqSubsampler(n_skip=args.skip_frame)(f["data/{}/states".format(key)].value)
        d_pos, _ = FixedFreqSubsampler(n_skip=args.skip_frame, aggregator=SumAggregator()) \
            (f["data/{}/right_dpos".format(key)].value, aggregate=True)
        d_quat, _ = FixedFreqSubsampler(n_skip=args.skip_frame, aggregator=QuaternionAggregator()) \
            (f["data/{}/right_dquat".format(key)].value, aggregate=True)
        gripper_actuation, _ = FixedFreqSubsampler(n_skip=args.skip_frame)(
            f["data/{}/gripper_actuations".format(key)].value)
        joint_velocities, _ = FixedFreqSubsampler(n_skip=args.skip_frame, aggregator=SumAggregator()) \
            (f["data/{}/joint_velocities".format(key)].value, aggregate=True)

        n_steps = states.shape[0]
        if args.target_length is not None and n_steps > args.target_length:
            continue

        # force the sequence of internal mujoco states one by one
        frames = []
        for i, state in enumerate(states):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            obs = env._get_observation()
            frame = obs["image"][::-1]
            frames.append(frame)

        frames = np.stack(frames, axis=0)
        actions = np.concatenate((d_pos, d_quat, gripper_actuation), axis=-1)

        pad_mask = np.ones((n_steps,)) if n_steps == args.target_length \
            else np.concatenate((np.ones((n_steps,)), np.zeros((args.target_length - n_steps,))))

        h5_path = os.path.join(args.output_path, "seq_{}.h5".format(key))
        with h5py.File(h5_path, 'w') as F:
            F['traj_per_file'] = 1
            F["traj0/images"] = frames
            F["traj0/actions"] = actions
            F["traj0/states"] = states
            F["traj0/pad_mask"] = pad_mask
            F["traj0/joint_velocities"] = joint_velocities


def steps2length(steps):
    return steps / (10 * 15)


def plot_stats(args, f):
    # plot histogram of lengths
    demos = list(f["data"].keys())
    lengths = []
    for key in tqdm.tqdm(demos):
        states = f["data/{}/states".format(key)].value
        lengths.append(states.shape[0])
    lengths = np.stack(lengths)
    fig = plt.figure()
    plt.hist(lengths, bins=30)
    plt.xlabel("Approx. Demo Length [sec]")
    # plt.title("Peg Assembly")
    # plt.xlim(5, 75)
    # plt.ylim(0, 165)
    fig.savefig(os.path.join(args.output_path, "length_hist.png"))
    plt.close()


class DataSubsampler:
    def __init__(self, aggregator):
        self._aggregator = aggregator

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This function needs to be implemented by sub-classes!")


class FixedFreqSubsampler(DataSubsampler):
    """Subsamples input array's first dimension by skipping given number of frames."""

    def __init__(self, n_skip, aggregator=None):
        super().__init__(aggregator)
        self._n_skip = n_skip

    def __call__(self, val, idxs=None, aggregate=False):
        """Subsamples with idxs if given, aggregates with aggregator if aggregate=True."""
        if self._n_skip == 0:
            return val, None

        if idxs is None:
            seq_len = val.shape[0]
            idxs = np.arange(0, seq_len - 1, self._n_skip + 1)

        if aggregate:
            assert self._aggregator is not None  # no aggregator given!
            return self._aggregator(val, idxs), idxs
        else:
            return val[idxs], idxs


class Aggregator:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This function needs to be implemented by sub-classes!")


class SumAggregator(Aggregator):
    def __call__(self, val, idxs):
        return np.add.reduceat(val, idxs, axis=0)


class QuaternionAggregator(Aggregator):
    def __call__(self, val, idxs):
        # quaternions get aggregated by multiplying in order
        aggregated = [val[0]]
        for i in range(len(idxs)-1):
            idx, next_idx = idxs[i], idxs[i+1]
            agg_val = val[idx][[3,0,1,2]]
            for ii in range(idx+1, next_idx):
                temp_val = val[ii][[3,0,1,2]]
                agg_val = self.quaternion_multiply(agg_val, temp_val)
            agg_val = agg_val[[1,2,3,0]]
            aggregated.append(agg_val)
        return np.asarray(aggregated)

    @staticmethod
    def quaternion_multiply(Q0, Q1):
        w0, x0, y0, z0 = Q0
        w1, x1, y1, z1 = Q1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                         x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                         -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                         x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


def actions2parts(actions):
    return actions[:, 0:3], actions[:, 3:7], actions[:, 7:8]


def parts2Actions(dpos_ds, dquat_ds, gripper_ds):
    return np.hstack((dpos_ds, dquat_ds, gripper_ds))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_folder", type=str,
                        default=os.path.join(robosuite.models.assets_root, "demonstrations/SawyerNutAssembly"))
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--skip_frame", type=int, default=0)
    parser.add_argument("--gen_dataset", type=bool, default=False)
    parser.add_argument("--plot_stats", type=bool, default=False)
    parser.add_argument("--target_length", type=int, default=-1)
    args = parser.parse_args()

    if args.target_length == -1:
        args.target_length = None

    # initialize an environment with offscreen renderer
    demo_file = os.path.join(args.demo_folder, "demo_eef.hdf5")
    f = h5py.File(demo_file, "r")
    env_name = f["data"].attrs["env"]
    env = make(
        env_name,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_height=args.height,
        camera_width=args.width,
    )

    # downsample

    # data:
    # states -> downsample
    # right dpos, dquat, gripper_actuations-> downsample with appropriate aggregrator
    # joint velocities -> downsample
    # images -> downsample
    # padmask -> downsample

    states = f["data/demo_1/states"].value
    eef_states = f["data/demo_1/eef_states"].value
    joint_velocities = f["data/demo_1/joint_velocities"].value
    dpos = f["data/demo_1/right_dpos"].value
    dquat = f["data/demo_1/right_dquat"].value
    gripper_actuation = f["data/demo_1/gripper_actuations"].value


    grp = f["data"]
    date, time, repository_version, env = grp.attrs["date"], grp.attrs["time"], grp.attrs["repository_version"], \
                                          grp.attrs["env"]

    f.close()

    # create a sum aggregator
    dpos_aggregator = SumAggregator()

    # create a multiply aggregator
    dquaternion_aggregator = QuaternionAggregator()

    # create a FixedFreqSubsampler object for normal downsampling
    fixed_freq_subsampler = FixedFreqSubsampler(args.skip_frame)

    # create a FixedFreqSubsampler object for dpos downsampling
    fixed_freq_subsampler_for_pos = FixedFreqSubsampler(args.skip_frame, dpos_aggregator)

    # create a FixedFreqSubsampler object for dquaternion downsampling
    fixed_freq_subsampler_for_quaternion = FixedFreqSubsampler(args.skip_frame, dquaternion_aggregator)

    # downsample states, images, padmask and joint velocities
    states_ds, _ = fixed_freq_subsampler(states)
    eef_states_ds, _ = fixed_freq_subsampler(eef_states)
    joint_velocities_ds, _ = fixed_freq_subsampler_for_pos(joint_velocities,aggregate=True)
    gripper_actuation_ds, _ = fixed_freq_subsampler(gripper_actuation)

    # find dpos, dquat, gripper using downsampled states
    dpos_ds = []
    dquat_ds = []


    for i in range(len(eef_states_ds)-1):
        cur_pos = eef_states_ds[i][:3]
        cur_quat = eef_states_ds[i][3:7]
        cur_gripper = eef_states_ds[i][7:]

        next_pos = eef_states_ds[i+1][:3]
        next_quat = eef_states_ds[i+1][3:7]
        next_gripper = eef_states_ds[i+1][7:]

        dpos_cur = next_pos-cur_pos
        dquat_cur = T.quat_multiply(T.quat_inverse(cur_quat), next_quat)
        gripper_cur = float(next_gripper-1)

        dpos_ds.append(dpos_cur)
        dquat_ds.append(dquat_cur)

    dpos_ds.append([0,0,0])
    dquat_ds.append([0,0,0,0])

    # store into demo_ds.hdf5 file
    demo_ds_file = os.path.join(args.demo_folder, "demo_ds.hdf5")
    f = h5py.File(demo_ds_file, "w")

    grp = f.create_group("data")

    ep_data_grp = grp.create_group("demo_1")
    ep_data_grp.attrs["model_file"] = "model_1.xml"
    ep_data_grp.create_dataset("states", data=np.array(states_ds))
    ep_data_grp.create_dataset("joint_velocities", data=np.array(joint_velocities_ds))
    ep_data_grp.create_dataset(
        "gripper_actuations", data=np.array(gripper_actuation_ds)
    )
    ep_data_grp.create_dataset("right_dpos", data=np.array(dpos_ds))
    ep_data_grp.create_dataset("right_dquat", data=np.array(dquat_ds))

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = robosuite.__version__
    grp.attrs["env"] = env_name

    f.close()




