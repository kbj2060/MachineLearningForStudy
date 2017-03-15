import numpy as np
import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import tensorflow as tf

def one_hot(x):
    return np.identity(16)[x: x+1]

register(
    id = 'FrozenLake-v3',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery' : True}
)

env = gym.make('FrozenLake-v3')
Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000

input_size = env.observation_space.n
output_size = env.action_space.n
lr = 0.1

X = tf.placeholder(tf.float32, shape = [1, input_size])
W = tf.Variable(tf.random_uniform([input_size,output_size],0,0.01))

Qpred = tf.matmul(X,W)
Y = tf.placeholder( tf.float32, shape = [1, output_size])

loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(lr ).minimize(loss)

dis = 0.99
num_ep = 2000
rList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in xrange(num_ep):
        s = env.reset()
        e = 1. /((i /50) + 10)
        rAll = 0
        done = False
        local_loss = []

        while not done:
            '''
             * Qs moves before Agent moves
            '''
            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})

            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            s1, reward, done, _ = env.step(a)

            '''
             * Qs is correct answer because if done, Qs get 0 or 1
             and if not done, Agent is survived so it gets points
             * Qs1 is advisor who leads us to the goal, but
               we are gonna trust him a little in this slippery game
            '''
            if done:
                Qs[0, a] = reward
            else:
                Qs1 = sess.run(Qpred, feed_dict={X : one_hot(s1)})
                Qs[0, a] = reward + dis * np.max(Qs1)

            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})
            rAll += reward
            s = s1
        print "loop is done!"
        rList.append(rAll)
    print("Percentage of successful episodes : " + str(sum(rList)/num_ep * 100)+ "%")
    plt.bar(range(len(rList)), rList, color="blue")
    plt.show()
