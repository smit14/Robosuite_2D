## Downsamplig sawyer trajectories

1. **states:**  
downsampled through state values. Works fine.

2. **joint velocities:**  
downsample the joint velocities. Aggregate the velocities while downsampling.
Works fine while relaying. Sometimes goes little far from original positions.

3. **end effector position and rotation change**  
downsample the dpos and dquat values of the end effector. Sum aggregate the dpos values and
quaternion aggregate the dquat values. Divert from the trajectory after a while (errors get
aggregated). 

4. **ik actions using downsampled states**  
generate ik actions(dpos and dquat) using the difference in the pos and quat of the
downsampled states. Even though generated ik actions are similiar to that of original 
ik actions, it doesn't follow the original trajectory while replaying.

 **Additional experiments**
 
 1. Changed the control frequency of environment. Resulted in difficulty to collect
 demonstration. Also this didn't affect the datasize.
 
 2. Changed the collection frequency in data collection wrapper. This reduced
 the datasize but can't replay original trajectory using joint velocities or
 ik controller.
 
 3. Tried to store data only when received any ik action from keyboard/spacemouse.
 It was not possible to collect demonstration using this(was difficult to control)
 
 
 



