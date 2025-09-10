# ROS integration of Precision-Focused Reinforcment Learning for Robotic Object Pushing

<img src="https://github.com/Grunex/ROS_precise_pushing/blob/main/assets/rosgraph.png" />

## Installation 
```bash
conda create -n precise_pushing
conda activate precise_pushing
cd PATH_TO_THIS_REPO
pip install -e .
```

## Running Simulation
Terminal 1
```bash
source ~/precise_pushing/rl-env/bin/activate
source ~/precise_pushing/catkin_ws/devel/setup.bash
roslaunch precise_pushing_bridge sim_bridge.launch env_id:=MujocoPandaPushEnv
```
Terminal 2
```bash
source ~/precise_pushing/rl-env/bin/activate
source ~/precise_pushing/catkin_ws/devel/setup.bash
rosrun precise_pushing_bridge policy_runner.py   _model_path:=/home/glx/precise_pushing/panda_push_data/rl/MujocoPandaPushEnv/evaluation/best_model.zip
```

## Acknowledgements
Special thanks to Lara Bergmann (@lbergmann1), David Leins, Robert Haschke, and Klaus Neumann for publicising their research paper.
https://github.com/ubi-coro/precise_pushing

## Maintainer
This repository is currently maintained by Pawel.
