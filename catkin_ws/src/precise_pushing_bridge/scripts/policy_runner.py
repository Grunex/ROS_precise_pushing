#!/usr/bin/env python3
import json
import time
import numpy as np
import rospy
import gymnasium as gym

from std_srvs.srv import Trigger
from std_msgs.msg import Float32MultiArray
from precise_pushing_bridge.srv import Step

from stable_baselines3 import SAC


def read_env_kwargs() -> dict:
    # Read ~env_kwargs JSON param (or return empty dict).
    s = rospy.get_param("~env_kwargs", "")
    if isinstance(s, str) and s.strip():
        return json.loads(s)  # raises if invalid
    return {}


def default_spec():
    return [
        {"key": "observation",   "length": 2, "shape": [2]},
        {"key": "achieved_goal", "length": 6, "shape": [6]},
        {"key": "desired_goal",  "length": 6, "shape": [6]},
    ]


class PolicyRunner:
    def __init__(self):
        self.ns = "/precise_pushing_bridge"
        self.model_path = rospy.get_param("~model_path", "")
        #if not self.model_path:
        #    raise RuntimeError("~model_path is required (path to SB3 .zip).")

        # Parameters
        self.deterministic = bool(rospy.get_param("~deterministic", True))
        self.rate_hz = float(rospy.get_param("~rate_hz", 120)) #high for responsive animation, can be lowered to 20
        self.env_id = rospy.get_param("~env_id", "MujocoPandaPushEnv")
        self.env_kwargs = read_env_kwargs()

        # ROS I/O 
        self.latest_flat = None  # will hold the last flat Float32 array from the bridge
        rospy.Subscriber(f"{self.ns}/obs/state", Float32MultiArray, self._on_flat, queue_size=1)
        self.reset_srv = rospy.ServiceProxy(f"{self.ns}/reset", Trigger) #reset
        self.step_srv  = rospy.ServiceProxy(f"{self.ns}/step",  Step)    #action

        # Dummy env (needed so SB3 can build the policy properly)
        rospy.loginfo(f"[runner] creating dummy env: {self.env_id}")
        self.dummy_env = gym.make(self.env_id, **self.env_kwargs)

        # Determine action length from the env action_space
        space = getattr(self.dummy_env, "action_space", None)
        if space is not None and getattr(space, "shape", None) is not None:
            self.action_len = int(np.prod(space.shape))
        else:
            self.action_len = 2  # safe fallback if the space is unknown

        # Load the SAC model on GPU
        rospy.loginfo(f"[runner] loading SAC model: {self.model_path}")
        self.model = SAC.load(self.model_path, env=self.dummy_env, device="cuda") #can also use device="cpu"

    # Callbacks & helpers
    def _on_flat(self, msg: Float32MultiArray):
        # Save the latest flat observation from the bridge # no queue
        self.latest_flat = np.asarray(msg.data, dtype=np.float32)

    def _flat_to_dict(self, flat: np.ndarray) -> dict:
        #Rebuild the dict observation for HER-trained policies.
        spec = default_spec()
        d = {}
        i = 0
        for item in spec:
            L = int(item["length"])
            shp = item.get("shape", [L])
            seg = flat[i:i + L]
            d[item["key"]] = seg.reshape(shp).astype(np.float32)
            i += L
        return d

    # Main loop
    def run(self):
        # 1 Wait for services to be ready
        rospy.loginfo("[runner] waiting for services…")
        self.reset_srv.wait_for_service()
        self.step_srv.wait_for_service()

        # 2 Reset the environment once
        rospy.loginfo("[runner] reset…")
        self.reset_srv()

        # 3 Wait for the first observation to arrive
        t0 = time.time()
        while self.latest_flat is None and (time.time() - t0) < 3.0 and not rospy.is_shutdown():
            time.sleep(0.05)

        # 4 Control loop
        rate = rospy.Rate(max(1.0, self.rate_hz))
        while not rospy.is_shutdown():
            if self.latest_flat is None:
                rate.sleep()
                continue

            # Build the observation dict
            obs_in = self._flat_to_dict(self.latest_flat)

            # Ask the model for an action
            action, _ = self.model.predict(obs_in, deterministic=self.deterministic)

            # Ensure the action is the right length and is a plain Python list
            a = np.asarray(action, dtype=np.float32).ravel()
            if a.size < self.action_len:
                a = np.pad(a, (0, self.action_len - a.size), mode="constant")
            elif a.size > self.action_len:
                a = a[:self.action_len]

            # Send the action to the bridge
            try:
                self.step_srv(a.tolist())
            except Exception as e:
                rospy.logerr(f"[runner] step failed: {e}")

            rate.sleep()


def main():
    rospy.init_node("policy_runner_sac")
    PolicyRunner().run()


if __name__ == "__main__":
    main()
