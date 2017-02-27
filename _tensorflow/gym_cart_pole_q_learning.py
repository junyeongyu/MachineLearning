import gym
import random
import numpy as np;
from gym import wrappers

env = gym.make('CartPole-v0')
#env = wrappers.Monitor(env, "tmp/gym-results")

observation_accuracy = 1
learning_rate = .6;

Q = dict()
available_actions = [0, 1]

done_count = 0;
for i_episode in range(300):
    diff = 0.0
    is_done = False
    observation = env.reset()
    for t in range(200):
        env.render()
        
        # 1. Build State
        state = map(lambda x: round(x,observation_accuracy), observation)
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
        
        # 4. do step & receive reward
        observation, reward, done, info = env.step(action)
        
        # 5. learn
        curr_diff = abs(np.sum(observation))
        reward = diff - curr_diff
        diff = curr_diff
        
        Q[state_key][action] += (learning_rate * (reward - Q[state_key][action]))
        
        #print observation;
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            is_done = True
            break
    if is_done == False:
        done_count += 1
        print("Episode {} COMPLETED".format(i_episode + 1) + "("+ str(done_count) + " - " + str(done_count * 100 / (i_episode + 1)) + "%)")

# 0.3 => Episode 500 COMPLETED(438 - 87%) - previous person. / Episode 500 COMPLETED(442 - 88%)
# 0.6 => Episode 500 COMPLETED(452 - 90%)
# 0.8 => Episode 500 COMPLETED(452 - 90%)

env.close()
#gym.upload("tmp/gym-results", api_key="sk_Z9gWx7pIQPO59bkfF8QoDg")