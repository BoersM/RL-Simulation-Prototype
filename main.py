import argparse
import time
from datetime import datetime
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from ple import PLE

from abstract_agent import ReplayMemory, loop_play_forever, q_loss
from agent import Agent
from simulation import CarSimulation, prepare_for_learning


def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default="run",
                        help='Mode to run. Either \'train\' a model or \'run\' using a model')
    parser.add_argument('-t', '--track', type=int, default=1, help='Track number, options are: [0, 1, 2]')
    parser.add_argument('-n', '--name', type=str, default="default",
                        help='The environment name for use in saving/loading')
    arguments = parser.parse_args()

    model_name = arguments.name
    mode = arguments.mode

    # training parameters
    num_epochs = 1
    num_steps_train = 15000  # steps per epoch of training
    num_steps_test = 3000
    update_frequency = 4  # step frequency of model training/updates

    # agent settings
    batch_size = 32
    num_frames = 4  # number of frames in a 'state'
    frame_skip = 2
    # percentage of time we perform a random action, help exploration.
    epsilon = 0.1
    epsilon_steps = 15000  # decay steps
    epsilon_min = 0.0001
    learning_rate = 0.01
    discount = 0.95  # discount factor
    rng = np.random.RandomState(24)

    # memory settings
    max_memory_size = 100000
    min_memory_size = 1000  # number needed before model training starts

    epsilon_rate = (epsilon - epsilon_min) / epsilon_steps

    rewards = {
        "tick": 0,
        "win": 100,
        "loss": -50
    }

    # PLE takes our game and the state_preprocessor. It will process the state
    # for our agent.
    game = CarSimulation(track_number=arguments.track)
    env = PLE(game, fps=60, reward_values=rewards)

    agent = Agent(env, batch_size, num_frames, frame_skip, learning_rate, discount, rng)

    filename = 'model_{}.h5'.format(arguments.name)
    model_file = Path(filename)

    if model_file.is_file():
        agent.load_model(filename, custom_objects={'q_loss': q_loss})
    else:
        agent.build_model()

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='/tmp/my_tf_logs',
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=True
    )
    tensorboard.set_model(agent.model)

    memory = ReplayMemory(max_memory_size, min_memory_size)

    env.init()

    if mode == "run":
        loop_play_forever(env, game, agent)
    else:
        training_rewards = []
        start_time = time.process_time()
        for epoch in range(1, num_epochs + 1):
            steps, num_episodes, batch_id = 0, 0, 0
            losses, rewards = [], []
            env.display_screen = False

            # training loop
            while steps < num_steps_train:
                episode_reward = 0.0
                agent.start_episode()

                while not env.game_over() and steps < num_steps_train:
                    state = prepare_for_learning(game.game_screen)
                    game.set_agent_information(episode=num_episodes, epoch=epoch)
                    reward, action = agent.act(state, epsilon=epsilon)
                    memory.add([state, action, reward, env.game_over()])

                    if steps % update_frequency == 0:
                        # print("Training on batch...")
                        loss = memory.train_agent_batch(agent)

                        if loss is not None:
                            tensorboard.on_batch_end(batch_id, named_logs(agent.model, [loss]))
                            losses.append(loss)
                            epsilon = max([epsilon_min, epsilon - epsilon_rate])
                            batch_id += 1

                    episode_reward += reward
                    steps += 1

                if num_episodes % 5 == 0:
                    print("Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward))

                if num_episodes % 20 == 0:
                    agent.save_model(filename)

                rewards.append(episode_reward)
                num_episodes += 1
                agent.end_episode()

            # Done training this epoch
            # Add the rewards to the total
            training_rewards.extend(rewards)
            print(
                "\nTrain Epoch {:02d}: Epsilon {:0.4f} | Avg. Loss {:0.3f} | Avg. Reward {:0.3f}".format(epoch, epsilon,
                                                                                                         np.mean(
                                                                                                             losses),
                                                                                                         np.sum(
                                                                                                             rewards) /
                                                                                                         num_episodes))

            steps, num_episodes = 0, 0
            losses, rewards = [], []

            # display the screen
            env.display_screen = True

            env.force_fps = False

            # testing loop
            while steps < num_steps_test:
                episode_reward = 0.0
                agent.start_episode()

                while not env.game_over() and steps < num_steps_test:
                    state = prepare_for_learning(game.game_screen)
                    game.set_agent_information(episode=num_episodes, epoch=epoch)
                    reward, action = agent.act(state, epsilon=0.05)

                    episode_reward += reward
                    steps += 1

                    # done watching after 500 steps.
                    if steps > 500:
                        env.force_fps = True
                        env.display_screen = False

                if num_episodes % 5 == 0:
                    print("Episode {:01d}: Reward {:0.1f}".format(num_episodes, episode_reward))

                rewards.append(episode_reward)
                num_episodes += 1
                agent.end_episode()

            print("Test Epoch {:02d}: Best Reward {:0.3f} | Avg. Reward {:0.3f}".format(epoch, np.max(rewards),
                                                                                        np.sum(rewards) / num_episodes))

        print("\nTraining took {} seconds".format(time.process_time() - start_time))
        print("\nTraining complete. Will loop forever playing!")
        plt.plot(training_rewards)
        plt.ylabel('reward')
        plt.savefig('rewards_{}_{}.png'.format(model_name, datetime.now().strftime('%Y%m%d%H%M%S')))
        tensorboard.on_train_end(None)
        loop_play_forever(env, game, agent)
