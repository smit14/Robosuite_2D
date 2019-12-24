# creating the world
from robosuite.models import MujocoWorldBase
world = MujocoWorldBase()

# creating the robot
from robosuite.models.robots import Sawyer
mujoco_robot = Sawyer()

# adding a gripper to robot Sawyer
from robosuite.models.grippers import gripper_factory
gripper = gripper_factory('TwoFingerGripper')
gripper.hide_visualization()
mujoco_robot.add_gripper("right_hand", gripper)

# add robot to the world
mujoco_robot.set_base_xpos([0,0,0])
world.merge(mujoco_robot)

# creating a table
from robosuite.models.arenas import TableArena
mujoco_arena = TableArena()
mujoco_arena.set_origin([0.16, 0, 0])
world.merge(mujoco_arena)

# adding the object
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_joint

object_mjcf = BoxObject()
world.merge_asset(object_mjcf)

obj = object_mjcf.get_collision(name="box_object", site=True)
obj.append(new_joint(name="box_object", type = "free"))
obj.set("pos", [0, 0, 0.5])
world.worldbody.append(obj)

# Simulation
model = world.get_model(mode="mujoco_py")

from mujoco_py import MjSim, MjViewer

sim = MjSim(model)
viewer = MjViewer(sim)
while True:
    sim.data.ctrl[:] = [1,2,3,4,5,6,7,8,9]
    sim.step()
    viewer.render()








