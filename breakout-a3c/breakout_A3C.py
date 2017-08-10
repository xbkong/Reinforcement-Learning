import gym

import time
import random
import threading
import multiprocessing

import keras as K
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers.convolutional import Conv2D

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

# Implementation of Asynchronous Advantage Actor Critic (A3C) for the breakout game (openai gym, breakout_v4)
global current_episode
current_episode = 0


class A3CGlobal:
    def __init__(self, possible_actions, inp_dims=(100, 100, 4), discount=0.99, lr_common=2.0e-4):
        self.state_dims = inp_dims
        self.discount = discount
        self.actor_lr = self.critic_lr = lr_common
        self.possible_actions = possible_actions
        self.num_threads = multiprocessing.cpu_count()
        self.actor, self.critic = self.create_model()
        self.optimizer = [self.actor_update_func(), self.critic_update_func()]
        self.session = tf.InteractiveSession()
        K.backend.set_session(self.session)
        self.session.run(tf.global_variables_initializer())

    # NN structure
    # Both actor and critic shares these initial layers: 2 * Conv(32, 8*8/4)[relu] -> FC(512)[relu]
    # Actor output: probabilities for each action
    # Critic output: value of state
    def create_model(self):
        inp = Input(shape=self.state_dims)
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inp)
        conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv1)
        fc1 = Dense(512, activation='relu')(Flatten()(conv2))
        policy = Dense(self.possible_actions, activation='softmax')(fc1)
        value = Dense(1, activation='linear')(fc1)

        actor = Model(inputs=inp, outputs=policy)
        actor._make_predict_function()
        critic = Model(inputs=inp, outputs=value)
        critic._make_predict_function()

        actor.summary()
        critic.summary()
        return actor, critic

    def train(self, save_interval=300):
        agents = []
        for i in range(self.num_threads):
            agents += [A3CWorker(self.possible_actions,
                                 self.state_dims,
                                 [self.actor, self.critic],
                                 self.session, self.optimizer,
                                 self.discount)]
        # Launch threads
        for agent in agents:
            time.sleep(0.3)
            agent.start()

        while True:
            time.sleep(save_interval)
            self.actor.save_weights("./breakout_actor.h5")
            self.critic.save_weights("./breakout_critic.h5")

    # Function for updating actor (actor loss + entropy)
    def actor_update_func(self):
        action = K.backend.placeholder(shape=[None, self.possible_actions])
        advantages = K.backend.placeholder(shape=[None, ])
        policy = self.actor.output
        smoothing_constant = 1e-10

        eligibility = K.backend.log(K.backend.sum(action * policy, axis=1) + smoothing_constant) * advantages
        loss = -K.backend.sum(eligibility)
        entropy = K.backend.sum(K.backend.sum(policy * K.backend.log(policy + smoothing_constant), axis=1))
        loss = loss + 0.01 * entropy
        optimizer = Adam(lr=self.actor_lr, epsilon=0.01)
        update = optimizer.get_updates(self.actor.trainable_weights, [], loss)

        return K.backend.function([self.actor.input, action, advantages], [loss], updates=update)

    # Function for updating critic (mean squared error)
    def critic_update_func(self):
        discounted_reward = K.backend.placeholder(shape=(None,))
        val = self.critic.output

        loss = K.backend.mean(K.backend.square(discounted_reward - val))
        optimizer = Adam(lr=self.critic_lr, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        return K.backend.function([self.critic.input, discounted_reward], [loss], updates=updates)


# make agents(local) and start training
class A3CWorker(threading.Thread):
    def __init__(self, possible_actions, state_dims, ac_models, session, update_funcs, discount):
        threading.Thread.__init__(self)

        self.possible_actions = possible_actions
        self.state_dims = state_dims
        self.actor, self.critic = ac_models
        self.session = session
        self.discount = discount
        self.update_funcs = update_funcs
        # Weight update interval
        self.update_counter = 0
        self.update_interval = 16

        self.states = list()
        self.actions = list()
        self.rewards = list()
        self.actor_, self.critic_ = self.create_model()

    # NN structure
    # Both actor and critic shares these initial layers: 2 * Conv(32, 8*8/4)[relu] -> FC(512)[relu]
    # Actor output: probabilities for each action
    # Critic output: value of state
    def create_model(self, summary=True):
        inp = Input(shape=self.state_dims)
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(inp)
        conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv1)
        fc1 = Dense(512, activation='relu')(Flatten()(conv2))
        policy = Dense(self.possible_actions, activation='softmax')(fc1)
        value = Dense(1, activation='linear')(fc1)

        actor = Model(inputs=inp, outputs=policy)
        actor._make_predict_function()
        critic = Model(inputs=inp, outputs=value)
        critic._make_predict_function()

        # Initialize with global weights
        actor.set_weights(self.actor.get_weights())
        critic.set_weights(self.critic.get_weights())

        if summary:
            actor.summary()
            critic.summary()

        return actor, critic

    def normalize(self, inp):
        return inp / 255.0

    def select_action(self, experience):
        experience = np.float32(self.normalize(experience))
        policy = self.actor_.predict(experience)[0]
        i = np.random.choice(self.possible_actions, 1, p=policy)[0]
        return i, policy

    def map_action(self, action):
        if action == 0 or action == 1:
            return action + 1
        else:
            return 3

    # Threaded run function
    def run(self, verbose=True):
        # Take global episode
        global current_episode
        env = gym.make("BreakoutDeterministic-v4")
        frame_no = 0
        while current_episode < 5000000:
            done = lost = False
            no_of_lives = 5
            score = 0
            current_observation = env.reset()
            observation_after = current_observation

            for _ in range(random.randint(1, 50)):
                current_observation = observation_after
                observation_after, _, _, _ = env.step(1)

            # Cut down input size to train faster
            state = np.maximum(observation_after, current_observation)
            state = np.uint8(resize(rgb2gray(state), (100, 100), mode='constant') * 255)
            experience = np.stack((state, state, state, state), axis=2)
            experience = np.reshape([experience], (1, 100, 100, 4))

            while not done:
                self.update_counter += 1
                frame_no += 1
                current_observation = observation_after
                action, policy = self.select_action(experience)
                # map action number to action number of gym
                gym_action = self.map_action(action)
                if lost:
                    action = 0
                    gym_action = 1
                    lost = False

                observation_after, reward, done, info = env.step(gym_action)
                next_observation = np.maximum(observation_after, current_observation)
                next_observation = np.uint8(resize(rgb2gray(next_observation), (100, 100), mode='constant') * 255)
                next_observation = np.reshape([next_observation], (1, 100, 100, 1))

                if no_of_lives > info['ale.lives']:
                    lost = True
                    no_of_lives = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1.0, 1.0)

                # Update states, actions and rewards with new interaction
                self.states.append(experience)
                a_0 = np.zeros(self.possible_actions)
                a_0[action] = 1
                self.actions.append(a_0)
                self.rewards.append(reward)

                experience_new = np.append(next_observation, experience[:, :, :, :3], axis=3)
                # reset experience if lost
                if lost:
                    experience = np.stack((next_observation,
                                           next_observation,
                                           next_observation,
                                           next_observation), axis=2)
                    experience = np.reshape([experience], (1, 100, 100, 4))
                else:
                    experience = experience_new

                # weight update
                if done or self.update_counter == self.update_interval:
                    self.train_(done)
                    self.critic_.set_weights(self.critic.get_weights())
                    self.actor_.set_weights(self.actor.get_weights())
                    self.update_counter = 0

                if done:
                    current_episode += 1
                    if verbose:
                        print "Episode:", current_episode, "  Frames:", frame_no, "  Score:", score
                    else:
                        print "Episode:", current_episode, "  Frames:", frame_no
                    frame_no = 0

    # Compute discounted rewards backwards
    def get_discounted_rewards(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        total = 0
        if not done:
            total = self.critic.predict(np.float32(self.normalize(self.states[-1])))[0]
        for t_i in range(len(rewards)-1, -1, -1):
            total = total * self.discount + rewards[t_i]
            discounted_rewards[t_i] = total
        return discounted_rewards

    # Train worker model
    def train_(self, done):
        discounted_rewards = self.get_discounted_rewards(self.rewards, done)
        states = np.array(self.states).reshape((len(self.states), 100, 100, 4))
        states = np.float32(self.normalize(states))

        values = self.critic.predict(states)
        values = np.reshape(values, len(values))

        actor_updater = self.update_funcs[0]
        critic_updater = self.update_funcs[1]
        actor_updater([states, self.actions, discounted_rewards - values])
        critic_updater([states, discounted_rewards])

        # Reset
        self.states = list()
        self.actions = list()
        self.rewards = list()


def main():
    a3c_model = A3CGlobal(possible_actions=3)
    a3c_model.train()

if __name__ == "__main__":
    main()
