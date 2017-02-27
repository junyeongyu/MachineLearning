import gym
from gym import wrappers

#env = wrappers.Monitor(env, "tmp/gym-results")


import gym
env = gym.make('CartPole-v0')

print env.action_space # number of input
print env.observation_space # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
print env.observation_space.high
print env.observation_space.low

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample() # take a random action
        #action = t % 2;
        #action = 1
        
        observation, reward, done, info = env.step(action)
        print observation;
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break



#env.close()
#gym.upload("tmp/gym-results", api_key="YOUR_API_KEY")