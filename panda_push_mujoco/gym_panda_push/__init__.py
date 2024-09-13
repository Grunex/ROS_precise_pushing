from gymnasium.envs.registration import register

def register_gymnasium_envs():
    #########################
    # MuJoCo                #
    #########################
    register(
        id='MujocoPandaPushSimpleEnv',
        entry_point='panda_push_mujoco.gym_panda_push.envs.mujoco.panda_push_simple_env:MuJoCoPandaPushSimpleEnv',
        max_episode_steps=50
    )

    register(
        id='MujocoPandaPushEnv',
        entry_point='panda_push_mujoco.gym_panda_push.envs.mujoco.panda_push_env:MuJoCoPandaPushEnv',
        max_episode_steps=50,
    )