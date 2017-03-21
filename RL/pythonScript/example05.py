import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque
import gym

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

dis = 0.9
REPLAY_MEMORY = 50000

def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0,action] = reward
        else:
            Q[0,action] = reward + dis * np.max(targetDQN.predict(next_state))

        x_stack = np.vstack([y_stack, Q])
        y_stack = np.vstack([x_stack, state])

    return mainDQN.update(x_stack, y_stack)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars,dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def bot_play(mainDQN):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if done:
            print "Total score : {}".format(reward_sum)
            break

def main():
    max_ep = 5000
    replay_buffer = deque()

    with tf.session() as sess:
        mainDQN = dqn.DQN(sess, input_size,output_size, name="main")
        targetDQN = dqn.DQN(sess, input_size,output_size, name="target")

        cp_op = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

        sess.run(cp_op)

        for episode in range(max_ep):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))

                next_state, reward, done, _ = env.step(action)
                if done:
                    reward = -100
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                state = next_state
                step_count += 1
                if step_count > 10000:
                    break

                print "episode : {} step : {}".format(episode, step_count)
                if step_count > 1000
