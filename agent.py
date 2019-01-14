from keras.layers import Convolution2D, Activation, Flatten, Permute
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from abstract_agent import AbstractAgent, q_loss


class Agent(AbstractAgent):

    def __init__(self, *args, **kwargs):
        AbstractAgent.__init__(self, *args, **kwargs)

    def build_model(self):
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=self.state_shape))
        model.add(Convolution2D(32, 10, 10, subsample=(4, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.num_actions))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        self.model = model
        print(model.summary())
