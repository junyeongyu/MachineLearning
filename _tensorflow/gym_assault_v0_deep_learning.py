import gym
import sys
import numpy as np
import copy

from gym import wrappers

sys.path.append("./agent/")
from dqn import DQNAgent

env = gym.make('Assault-v0')

#env = wrappers.Monitor(env, "tmp/gym-results", force=True)

print env.action_space # number of input
print env.observation_space # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
#print env.observation_space.high
#print env.observation_space.low

agent = DQNAgent(env)
agent.load("./save/assault-v0.h5")
for i_episode in range(10000):
    observation = env.reset()
    state = np.reshape(observation, [1, 120000])
    t = 0;
    score = 0;
    while True :
        t += 1;
        env.render()
        
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 120000])
        score += reward;
        #reward = score;
        reward = -1 * (score / 5) if done else reward
        agent.remember(state, action, reward, next_state)
        state = copy.deepcopy(next_state)
       
        if done:
            print("{} Episode finished after {} timesteps (score: {}, e: {})".format(i_episode + 1, t, score, agent.epsilon));
            break
    if i_episode % 5 == 0:
        agent.save("./save/assault-v0.h5")
    agent.replay(5);

env.close()
#gym.upload("tmp/gym-results", api_key="sk_Z9gWx7pIQPO59bkfF8QoDg")