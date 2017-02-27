import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

import gym
env = gym.make('CartPole-v0')

env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    #env.render()
    observation, reward, done, _ = env.step(np.random.randint(0,2))
    reward_sum += reward
    if done:
        random_episodes += 1
        print ("Reward for this episode was:",reward_sum)
        reward_sum = 0
        env.reset()
        


H = 10 # 은닉층의 노드 수
batch_size = 5 # 몇개의 에피소드마다 파라미터를 업데이트할 것인지
learning_rate = 1e-2 # 학습률
gamma = 0.99 # 보상에 대한 할인 인자

D = 4 # 입력 차원

tf.reset_default_graph()

observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
# W1은 은닉층으로 보낸다
W1 = tf.get_variable("W1", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
# relu 활성화함수를 쓴다
layer1 = tf.nn.relu(tf.matmul(observations,W1))
# 은닉층의 결과인 10개의 값으로 하나의 결과값(점수)을 낸다
W2 = tf.get_variable("W2", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
# 점수를 확률로 변환한다.
probability = tf.nn.sigmoid(score)

# 학습 가능한 변수들 (가중치)
tvars = tf.trainable_variables()
# 출력값을 받는 부분
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
# 이득을 받는 부분
advantages = tf.placeholder(tf.float32,name="reward_signal")

# 손실함수. 좋은 이득(시간 경과에 따른 보상)을 더 자주 주는 행동으로 
# 가중치를 보내고, 덜 가능성이 있는 행동에 가중치를 보낸다.

# 왜 이게 동작하는 것일까?
# cross entropy와 비슷하다. 내가 행동을 1로 했고, 그 행동에 높은 확률을 주었다면 손실이 작고,
# 내가 0으로 움직였고, 그 행동에 낮은 확률을 주었다면 손실이 작다.
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
# 위의 각 행동의 잘하고 못하고 부분을 지연된 보상으로 조정하고 난 모든 것을 손실로 본다.
loss = -tf.reduce_mean(loglik * advantages) 
# 이 손실을 이용해 학습 변수들의 그라디언트를 구한다.
newGrads = tf.gradients(loss,tvars)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
# 여러 에피소드로부터의 그라디언트를 모았다가 그것을 적용한다.
# 왜 매 에피소드마다 그라디언트를 업데이트하지 않느냐면 에피소드의 노이즈까지 학습할까봐
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # 최적화기 adam
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # 그라디언트 저장하는 부분
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
# 그라디언트 적용하는 부분
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

