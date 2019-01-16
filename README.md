# RL-Simulation-Prototype

Reinforcement learning using a car simulation. Based on Tensorflow/Keras and Pygame Learning Environment.

## Getting Started

### Prerequisites

It is recommended to install anaconda and use the included environment.yml to set up your environment. Make sure to manually install PLE.

### How to run

The application can be run in either a training or run mode. There is also an option for a track type [0, 1, 2]. Also if no name is specified the model files will be generated using the default value. This will override any model file with the same name so be aware!

Train:
```
python main.py -m train -t 0 -n test
```

Run (a model file needs to be available):
```
python main.py -m run -t 0 -n test
```

A blank screen will come up. This is intended as drawing the environment slows learning. Once learning is done the environment will be drawn. There will be logging in regards to the status every five episodes.

## Built With

PLE
Tensorflow
Keras

## Authors

Mark Boers

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* PLE for providing an excellent interface.
