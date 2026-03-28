from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv('citylearn_challenge_2020_climate_zone_1')

obs = env.reset()

for _ in range(10):
    actions = [space.sample() for space in env.action_space]
    obs, reward, done, info = env.step(actions)

    total_reward = sum(reward)
    print("Total reward:", total_reward)