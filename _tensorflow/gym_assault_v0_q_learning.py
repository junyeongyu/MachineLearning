import gym
import random
import sys
import numpy as np
import copy
import pickle

from gym import wrappers

sys.path.append("./agent/")
from dqn import DQNAgent

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)
        
env = gym.make('Assault-v0')

#env = wrappers.Monitor(env, "tmp/gym-results", force=True)

print env.action_space # number of input
print env.observation_space # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
#print env.observation_space.high
#print env.observation_space.low

observation_accuracy = 1 #QL
learning_rate = .6; #QL
Q = dict() #QL
available_actions = [0, 1, 2, 3, 4, 5, 6] #QL

agent = DQNAgent(env)
agent.load("./save/assault-v0.h5")
Q = load_object("./save/assault-v0.pkl") #QL
for i_episode in range(10000):
    observation = env.reset()
    #state = np.reshape(observation, [1, 120000])
    t = 0;
    score = 0;
    while True :
        t += 1;
        #env.render()
        
        
        #action = agent.act(state)
        #next_state, reward, done, _ = env.step(action)
        #next_state = np.reshape(next_state, [1, 120000])
        #score += reward;
        #agent.remember(state, action, reward, next_state)
        #state = copy.deepcopy(next_state)
        
        # 1. Build State
        state = np.reshape(observation, [1, 120000]); #map(lambda x: round(x,observation_accuracy), observation)
        state_key = ' '.join([str(x) for x in state])
        #print state_key
        
        # 2. Create Q entry for state
        if not state_key in Q:
            Q[state_key] = dict()
            for action in available_actions:
                Q[state_key][action] = 0.0
        
        # 3. Choose action
        maxQ = -10000.0
        for action in Q[state_key].keys():
            if Q[state_key][action] > maxQ:
                maxQ = Q[state_key][action]
        # add all actions with maxQ to actions-list
        maxQ_actions = [];
        for action in Q[state_key].keys():
            if Q[state_key][action] == maxQ:
                maxQ_actions.append(action)
        
        # randomly choose one of the maxQ actions
        action = maxQ_actions[random.randint(0, len(maxQ_actions) - 1)]
        
        #print action
        
        # 4. do step & receive reward
        observation, reward, done, info = env.step(action)
        score += reward;
        
        # 5. learn
        Q[state_key][action] += (learning_rate * (reward - Q[state_key][action]))
        
        if done:
            print("{} Episode finished after {} timesteps (score: {})".format(i_episode + 1, t, score));#, agent.epsilon));
            break
    #if i_episode % 5 == 0:
    #    agent.save("./save/assault-v0.h5")
    #agent.replay(3);
    if i_episode % 10 == 0:
        save_object(Q, "./save/assault-v0.pkl") #QL

env.close()
#gym.upload("tmp/gym-results", api_key="sk_Z9gWx7pIQPO59bkfF8QoDg")