import gymnasium as gym
import os
import numpy as np


def simulate(solution, seed=None, env=None, video_dir=None, episode_id=0):
    # check if env provided, else create env
    if not env:
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
    # check if recording video
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix=f"lander_{episode_id}"
        )

    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    solution = solution.reshape((action_dim, obs_dim))

    # initialise values
    total_reward = 0.0
    obs, _ = env.reset(seed=seed)
    done = False
    impact_x_pos = None
    impact_x_vel = None
    impact_y_vel = None
    all_x_vels = []
    all_y_vels = []

    # while lander session has not ended (not truncated or terminated)
    while not done:
        # obtain best action for current state (observation)
        action = np.argmax(solution @ obs)
        # evaluate action
        obs, reward, terminated, truncated, _ = env.step(action)
        # update values
        done = terminated or truncated
        total_reward += reward
        x_pos = obs[0]
        x_vel = obs[2]
        y_vel = obs[3]
        leg0_touch = obs[6]
        leg1_touch = obs[7]
        all_y_vels.append(y_vel)
        all_x_vels.append(x_vel)

        #  if lunar makes contact w the ground for the first time, update impact values
        if not impact_x_pos and (leg0_touch or leg1_touch):
            impact_x_pos = x_pos
            impact_x_vel = x_vel
            impact_y_vel = y_vel

    # if session ended without end conditions
    if impact_x_pos is None:
        impact_x_pos = x_pos
        impact_y_vel = min(all_y_vels)
        impact_x_vel = max(all_x_vels, key=abs)

    env.close()

    return total_reward, impact_y_vel, impact_x_pos, impact_x_vel, 