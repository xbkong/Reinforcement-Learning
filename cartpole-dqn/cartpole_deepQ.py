import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np
from collections import deque


# Deep Q Network with experience replay and target network
class DeepQLearner:
    def __init__(self, state_size, action_size, train_size=2000, batch_size=64, gamma=0.99, lr=0.001, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory = deque(maxlen=train_size)
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        # Network structure: Dense(32) -> Dense(32) -> Dense(2)
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_normal'))
        model.add(Dense(20, activation='relu', kernel_initializer='glorot_normal'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='glorot_normal'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def pick_action(self, state):
        rand = random.random()
        if rand > self.epsilon:
            return np.argmax(self.model.predict(state)[0])
        else:
            return random.randrange(self.action_size)

    def train(self):
        # Not enough samples, do not proceed
        if len(self.memory) < 1000:
            return
        batch = random.sample(self.memory, self.batch_size)

        inp = np.zeros((self.batch_size, self.state_size))
        update_target = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            inp[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            update_target[i] = batch[i][3]
            done.append(batch[i][4])

        target = self.model.predict(inp)
        q_values = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(q_values[i]))

        self.model.fit(inp, target, batch_size=self.batch_size, epochs=1, verbose=0)


def run(version=1, render=True, verbose=True):
    if version != 1 and version != 0:
        print "CartPole Version Not Supported"
        return
    env = gym.make('CartPole-v0')
    maxscore = 200
    if version == 0:
        env._max_episodes = 500
    if version == 1:
        env = gym.make('CartPole-v1')
        maxscore = 500

    iterations = 500

    # Setup scores, state size, action size
    scores = []
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    learner = DeepQLearner(state_size, action_size)

    for i in range(iterations):
        done = False
        score = 0
        state = env.reset()
        # Flatten first
        state = np.reshape(state, [1, state_size])
        # Do game play until done
        while done == False:
            if render:
                env.render()
            # pick an action following an epsilon greedy strategy
            action = learner.pick_action(state)
            # Get interaction from env, skip info as it is not used
            next_state, reward, done, _ = env.step(action)

            # Save the state, action, reward next_state and done tuple to memory for training
            learner.memory.append((state, action, reward, next_state, done))
            # Decrease epsilon from 1 so that more exploration happens in the beginning
            if learner.epsilon > 0.01:
                learner.epsilon *= 0.999
            learner.train()

            next_state = np.reshape(next_state, [1, state_size])
            score += reward
            state = next_state

            if done:
                # Update model, score, track scores and display
                learner.target_model.set_weights(learner.model.get_weights())
                scores.append(score)
                if verbose:
                    print "Iteration:", i, "  score:", score, "epsilon:", learner.epsilon
                else:
                    print "Iteration:", i, "  score:", score

                # Terminate training once the last 5 game plays all got near perfect score
                if len(scores) > 5 and np.mean(scores[-5:]) >= maxscore-1:
                    return

if __name__ == "__main__":
    run()
