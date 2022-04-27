import gym
import gym_grasp # This includes GraspBlock-v0

gym.logger.set_level(40)

env = gym.make('GraspObject-v0')

env.reset()
while True:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

    env.render()
    if done:
        obs = env.reset()
