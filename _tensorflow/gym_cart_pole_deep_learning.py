import gym
import random
import copy
import numpy as np

from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

# https://keon.io/rl/deep-q-learning-with-keras-and-gym/
# Deep-Q learning Agent
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.memory = []
        self.gamma = 0.9  # decay rate
        self.epsilon = 0 # 1  # exploration
        self.epsilon_decay = .995
        self.epsilon_min = 0.1
        self.learning_rate = 0.0001
        self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=4, activation='tanh'))
        model.add(Dense(128, activation='relu')) #tanh
        model.add(Dense(128, activation='relu')) #tanh
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        self.model = model
    
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def replay(self, batch_size):
        batches = min(batch_size, len(self.memory))
        batches = np.random.choice(len(self.memory), batches)
        for i in batches:
            state, action, reward, next_state = self.memory[i]
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def load(self, name):
        self.model.load_weights(name)
    def save(self, name):
        self.model.save_weights(name)

env = gym.make('CartPole-v0')
agent = DQNAgent(env)
agent.load("./save/cartpole.h5")

episodes = 10000
done_count = 0
for i_episode in range(episodes):
    diff = 0.0
    is_done = False
    observation = env.reset()
    state = np.reshape(observation, [1, 4])
    for t in range(200):
        #env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        reward = -100 if done else reward * 1
        agent.remember(state, action, reward, next_state)
        state = copy.deepcopy(next_state)
        
        if done:
            print("episode: {}/{}, score: {}, memory size: {}, e: {}".format(i_episode, episodes, t, len(agent.memory), agent.epsilon))
            is_done = True
            break
    if i_episode % 50 == 0:
        agent.save("./save/cartpole.h5")
    agent.replay(32);
    
    if is_done == False:
        done_count += 1
        print("Episode {} COMPLETED".format(i_episode + 1) + "("+ str(done_count) + " - " + str(done_count * 100 / (i_episode + 1)) + "%)")

env.close()