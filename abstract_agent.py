import json
from collections import deque

import h5py
import numpy as np
import tensorflow as tf
from keras.engine.saving import _deserialize_model
from keras.optimizers import SGD, Adam, RMSprop
# noinspection PyMethodMayBeStatic
from keras.utils import h5dict

from simulation import prepare_for_learning


def q_loss(y_true, y_pred):
    # assume clip_delta is 1.0
    # along with sum accumulator.
    diff = y_true - y_pred
    _quad = tf.minimum(abs(diff), 1.0)
    _lin = abs(diff) - _quad
    loss = 0.5 * _quad ** 2 + _lin
    loss = tf.reduce_sum(loss)

    return loss


class AbstractAgent:
    """
        DQN agent with replay memory and decaying epsilon greedy.
        Based on https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/examples/example_support.py
    """

    def __init__(self, env, batch_size, num_frames, frame_skip, learning_rate, discount, rng, optimizer="adam",
                 frame_dim=None):
        self.env = env
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.learning_rate = learning_rate
        self.discount = discount
        self.rng = rng

        if optimizer == "adam":
            opt = Adam(lr=self.learning_rate)
        elif optimizer == "sgd":
            opt = SGD(lr=self.learning_rate)
        elif optimizer == "sgd_nesterov":
            opt = SGD(lr=self.learning_rate, nesterov=True)
        elif optimizer == "rmsprop":
            opt = RMSprop(lr=self.learning_rate, rho=0.9, epsilon=0.003)
        else:
            raise ValueError("Unrecognized optimizer")

        self.optimizer = opt

        self.frame_dim = (100, 100)
        self.state_shape = (num_frames,) + self.frame_dim
        self.input_shape = (batch_size,) + self.state_shape

        self.state = deque(maxlen=num_frames)
        self.actions = self.env.getActionSet()
        self.num_actions = len(self.actions)
        self.model = None

    def build_model(self):
        pass

    def load_model(self, filepath, custom_objects=None, compile_model=True):
        if h5py is None:
            raise ImportError('`load_model` requires h5py.')
        opened_new_file = not isinstance(filepath, h5py.Group)
        f = h5dict(filepath, 'r')
        try:
            model = _deserialize_model(f, custom_objects, compile_model)
        finally:
            if opened_new_file:
                f.close()
        self.model = model

    def save_model(self, filename):
        print("Now we save the model")
        self.model.save(filename)
        with open(filename[:-3] + ".json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)

    def predict_single(self, state):
        """
            model is expecting a batch_size worth of data. We only have one states worth of
            samples so we make an empty batch and set our state as the first row.
        """
        states = np.zeros(self.input_shape)
        states[0, ...] = state.reshape(self.state_shape)

        return self.model.predict(states)[0]  # only want the first value

    def _argmax_rand(self, arr):
        # picks a random index if there is a tie
        return self.rng.choice(np.where(arr == np.max(arr))[0])

    def _best_action(self, state):
        q_vals = self.predict_single(state)
        # print("q_vals: {}, argmax: {}".format(q_vals, str(np.argmax(q_vals))))
        return np.argmax(q_vals)  # the action with the best Q-value

    def act(self, state, epsilon=1.0):
        self.state.append(state)

        action = self.rng.randint(0, self.num_actions)
        if len(self.state) == self.num_frames:  # we havent seen enough frames
            _state = np.array(self.state)

            if self.rng.rand() > epsilon:
                action = self._best_action(_state)  # exploit
            # else:
            #     print("Random action taken...")

        reward = 0.0
        for i in range(self.frame_skip):  # we repeat each action a few times
            # act on the environment
            reward += self.env.act(self.actions[action])

        reward = np.clip(reward, -1.0, 1.0)

        return reward, action

    def start_episode(self, n=3):
        self.env.reset_game()  # reset
        for i in range(self.rng.randint(n)):
            self.env.act(self.env.NOOP)  # perform a NOOP

    def end_episode(self):
        self.state.clear()


class ReplayMemory:

    def __init__(self, max_size, min_size):
        self.min_replay_size = min_size
        self.memory = deque(maxlen=max_size)

    def __len__(self):
        return len(self.memory)

    def add(self, transition):
        self.memory.append(transition)

    def train_agent_batch(self, agent):
        if len(self.memory) > self.min_replay_size:
            states, targets = self._random_batch(agent)  # get a random batch
            return agent.model.train_on_batch(states, targets)
        else:
            return None

    def _random_batch(self, agent):
        inputs = np.zeros(agent.input_shape)
        targets = np.zeros((agent.batch_size, agent.num_actions))

        seen = []
        idx = agent.rng.randint(0, high=len(self.memory) - agent.num_frames - 1)

        for i in range(agent.batch_size):
            while idx in seen:
                idx = agent.rng.randint(0, high=len(self.memory) - agent.num_frames - 1)

            states = np.array([self.memory[idx + j][0] for j in range(agent.num_frames + 1)])
            art = np.array([self.memory[idx + j][1:] for j in range(agent.num_frames)])

            actions = art[:, 0].astype(int)
            rewards = art[:, 1]
            terminals = art[:, 2]

            state = states[:-1]
            state_next = states[1:]

            inputs[i, ...] = state.reshape(agent.state_shape)
            # we could make zeros but pointless.
            targets[i] = agent.predict_single(state)
            q_prime = np.max(agent.predict_single(state_next))

            targets[i, actions] = rewards + (1 - terminals) * (agent.discount * q_prime)

            seen.append(idx)

        return inputs, targets


def loop_play_forever(env, game, agent):
    # our forever play loop
    try:
        # slow it down
        env.display_screen = False
        env.force_fps = False
        num_episodes = 0
        while True:
            agent.start_episode()
            episode_reward = 0.0
            while not env.game_over():
                state = prepare_for_learning(game.game_screen)
                game.set_agent_information(episode=num_episodes)
                reward, action = agent.act(state, epsilon=0.05)
                episode_reward += reward

            print("Agent scored {:0.1f} reward for episode.".format(episode_reward))
            num_episodes += 1
            agent.end_episode()

    except KeyboardInterrupt:
        print("Exiting out!")
