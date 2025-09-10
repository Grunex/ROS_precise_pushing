#!/usr/bin/env python3
import json
import numpy as np
import rospy
import gymnasium as gym

from std_srvs.srv import Trigger, TriggerResponse
from precise_pushing_bridge.srv import Step, StepResponse
from std_msgs.msg import Float32MultiArray


def read_env_kwargs():
    s = rospy.get_param("~env_kwargs", "")
    if isinstance(s, str) and s.strip():
        return json.loads(s)  # will throw if invalid
    return {}


#necessary for ROS
def flatten_obs(obs):
    #Dict -> concat values in key order; else -> ravel to 1D.
    if isinstance(obs, dict):
        parts = []
        for k in obs.keys():
            v = obs[k]
            a = np.asarray(v)
            parts.append(a.ravel())
        return np.concatenate(parts).astype(np.float32) if parts else np.array([], np.float32)
    return np.asarray(obs, np.float32).ravel()


class Bridge:
    def __init__(self):
        self.env_id = rospy.get_param("~env_id", "MujocoPandaPushEnv")
        self.env_kwargs = read_env_kwargs()

        rospy.loginfo(f"[bridge] creating env: {self.env_id} kwargs={self.env_kwargs}")
        self.env = gym.make(self.env_id, **self.env_kwargs)

        self.obs_pub = rospy.Publisher("~obs/state", Float32MultiArray, queue_size=1, latch=True)
        self.reset_srv = rospy.Service("~reset", Trigger, self.on_reset)
        self.step_srv  = rospy.Service("~step",  Step,    self.on_step)

        # just to avoid step-before-reset errors
        self._did_reset = False

    def on_reset(self, _):
        obs, _ = self.env.reset()
        self._did_reset = True
        self.obs_pub.publish(Float32MultiArray(data=flatten_obs(obs).tolist()))
        return TriggerResponse(success=True, message="Environment reset.")

    def on_step(self, req):
        if not self._did_reset:
            # auto-reset
            obs, _ = self.env.reset()
            self._did_reset = True
            self.obs_pub.publish(Float32MultiArray(data=flatten_obs(obs).tolist()))

        # numpy action
        a = np.array(req.action, dtype=np.float32)

        # shape/clip to env action space 
        try:
            space = self.env.action_space
            if getattr(space, "shape", None) is not None:
                need = int(np.prod(space.shape))
                if a.size < need:
                    a = np.concatenate([a, np.zeros(need - a.size, np.float32)], 0)
                elif a.size > need:
                    a = a[:need]
                a = a.reshape(space.shape)
            if hasattr(space, "low") and hasattr(space, "high"):
                a = np.clip(a, space.low, space.high)
            else:
                a = np.clip(a, -1.0, 1.0)
        except Exception:
            a = np.clip(a, -1.0, 1.0)

        obs, reward, terminated, truncated, _ = self.env.step(a)
        done = bool(terminated or truncated)

        self.obs_pub.publish(Float32MultiArray(data=flatten_obs(obs).tolist()))
        return StepResponse(reward=float(reward), done=done)


def main():
    rospy.init_node("precise_pushing_bridge")
    Bridge()
    rospy.spin()


if __name__ == "__main__":
    main()

