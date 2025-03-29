"""------------------------------------------Important!!!------------------------------------------"""
'''Please install specific version 0.21.0 of OpenAI Gym using command following:'''
'''pip uninstall gym'''
'''pip install gym==0.21.0'''

#Double Deep Q-Learning (DDQN):
import gym
import numpy as np
import random
from collections import deque
import time
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

"""
Double Deep Q-Learning (DDQN) Algorithm for:
    LunarLander_v2
"""

# Get the directory of the current Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define mode (True = No Training):
FLAG_PLAY_ONLY = True  #if FLAG_PLAY_ONLY is activated, the model will be loaded and the agent will play the game without training
#When FLAG_PLAY_ONLY = True, the program will not train the agent, but will simply play (or use the already trained model).
#When FLAG_PLAY_ONLY = False, the program will train the agent.
# These parameters are only a suggestion. Feel free to change them in order to achieve better and more stable results

# Minimum value for the exploration rate
EXP_RATE_MIN = 0.005
# decay rate for the ϵ exploration rate
DECAY_RATE = 0.99 #ϵnew=ϵold×0.99
# Maximum number of training episodes
MAX_TRAINING_EPISODES = 2000
MAX_TRAINING_STEPS = 1000
# Maximum size of the buffer to store transitions
MAX_SIZE_BUFFER = 500000
# Minibatch size used for training of the neural net
MINIBATCH_SIZE = 64
# Learning rate used for training of the neural net
LEARNING_RATE = 1e-3
# Number of steps after which to update the second neural net
UPDATE_TARGET_AFTER_STEPS = 2000
# Discount factor used  for training of the neural net
GAMMA = 0.99
#If GAMMA ≈ 1 (e.g., 0.99), the agent assigns high importance to future rewards, aiming for long-term benefits.
#If GAMMA ≈ 0 (e.g., 0.1), the agent is focused on short-term rewards.

# Necessary mean success to consider task as solved
MEAN_SUCCESS = 200 #at which the task is considered solved
# Number of episodes regarded to compute the mean reward
MEAN_REWARD_EPISODES = 100


def convert_time(time_in): #for converting time in seconds to hours, minutes and seconds
    # Input:
    #   - time_in: float, time in seconds
    # Return:
    #   - hours: int
    #   - minutes: int
    #   - seconds: int
    # Function:
    #   - convert seconds to hours, minutes and seconds
    #   - e.g. convert_time(3666) = (1, 1, 6)

    ### Enter your code here ###
    hours = int(time_in // 3600)
    minutes = int((time_in % 3600) // 60)
    seconds = int(time_in % 60)
    ### End of your code ###

    return hours, minutes, seconds


class LearningStats:
    def __init__(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - define empty list for rewards, mean_rewards and steps

        ### Enter your code here ###
        self.rewards = []
        self.mean_rewards = []
        self.steps = []
        ### End of your code ###

    def append_step(self, reward, steps):
        # Input:
        #   - self
        #   - reward: reward of episode
        #   - steps: number of steps of episode
        # Return:
        #   - none
        # Function:
        #   - Compute mean reward for last MEAN_REWARD_EPISODES episodes
        #   - If less than MEAN_REWARD_EPISODES have been trained so far, set the mean reward to 0
        #   - Append reward, steps and mean reward to corresponding list

        ### Enter your code here ###
        self.rewards.append(reward)
        self.steps.append(steps)
        mean_reward = np.mean(self.rewards[-MEAN_REWARD_EPISODES:]) if len(self.rewards) >= MEAN_REWARD_EPISODES else 0
        self.mean_rewards.append(mean_reward)
        ### End of your code ###

    def plot_stats(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - Plot reward per episode, mean reward, steps and a horizontal line to mark the necessary mean reward,
        #     which has to be reached to consider the task as solved
        #   - Make sure, that the mean reward and the horizontal line are not covered by the line of reward per episode
        #     Hint: See task sheet for example and use 'zorder'
        #   - Add labels to the axes and a legend

        ### Enter your code here ###
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards, label="Reward per Episode", zorder=1)
        plt.plot(self.mean_rewards, label="Mean Reward", zorder=2)
        plt.axhline(y=MEAN_SUCCESS, color='r', linestyle='--', label="Success Threshold", zorder=3)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()
        ### End of your code ###


class Network:
    def __init__(self, num_obs, num_act, gamma=GAMMA, lr=LEARNING_RATE):
        # Input:
        #   - self
        #   - num_obs: Number of observations of environment
        #   - num_act: Number of actions available in the environment
        #   - gamma: discount factor
        #   - lr: learning rate used for training of the network
        # Return:
        #   - none
        # Function:
        #   - Define variables for number of observations and actions, the discount factor and the learning rate
        #   - Define loss function (Huber loss) and optimizer (Adam, use specified learning rate)
        #   - Create a replay buffer with maximum size of MAX_SIZE_BUFFER
        #     (use deque here, as it automatically overwrites old entries, if maximum length of buffer is reached)
        #   - Define model and target model for double Deep-Q-Learning
        #   - Create a variable to count the training steps since the net was updated, set it to 0

        ### Enter your code here ###
        self.num_obs = num_obs
        self.num_act = num_act
        self.gamma = gamma
        self.lr = lr
        self.loss = tf.keras.losses.Huber()
        self.opt = Adam(learning_rate=self.lr)
        self.replay_buffer = deque(maxlen=MAX_SIZE_BUFFER)
        self.model = self.create_net()
        self.target_model = self.create_net()
        self.target_model.set_weights(self.model.get_weights())
        self.training_steps = 0
        ### End of your code ###

    def create_net(self):
        # Input:
        #   - self
        # Return:
        #   - model: Keras model of neural net
        # Function:
        #   - create the neural net: specify the layers and its optimizer and loss function
        #   - Hint: Use 2 hidden layers with 128 and 64 neurons, use ReLu and a dropout of 0.2

        ### Enter your code here ###
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.num_obs,)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(self.num_act, activation='linear')
        ])
        model.compile(optimizer=self.opt, loss=self.loss)
        return model
        ### End of your code ###

    def prepare_update(self, update_target_after_steps=UPDATE_TARGET_AFTER_STEPS):
        # Input:
        #   - self
        #   - steps: total number of steps in training process so far
        #   - update_target_after_steps: Number of steps after which target model is updated
        # Return:
        #   - none
        # Function:
        #   - Sample a minibatch from replay buffer and save it in transitions
        #   - Update target network if necessary

        ### Enter your code here ###
        if len(self.replay_buffer) < MINIBATCH_SIZE:
            return
        transitions = random.sample(self.replay_buffer, MINIBATCH_SIZE)
        if self.training_steps % update_target_after_steps == 0:
            self.target_model.set_weights(self.model.get_weights())
        self.training_steps += 1
        ### End of your code ###

        # DO NOT TOUCH ANYTHING HERE!

        idx = np.array([i for i in range(MINIBATCH_SIZE)])
        states = [i[0] for i in transitions]
        actions = [i[1] for i in transitions]
        next_states = [i[2] for i in transitions]
        rewards = [i[3] for i in transitions]
        dones = [i[4] for i in transitions]

        # Convert inputs from mini_batch to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        idx = tf.convert_to_tensor(idx, dtype=tf.int32)

        # separate function which is defined as tf.function to improve computing speed
        self.update(states, rewards, actions, dones, next_states, idx)

    @tf.function
    def update(self, states, rewards, actions, dones, next_states, idx):

        # DO NOT TOUCH ANYTHING HERE!

        # compute forward pass for next states
        target_q_sel = self.model(next_states)
        # get next action
        next_action = tf.argmax(target_q_sel, axis=1)

        # apply double Deep-Q-Learning
        # compute forward pass for next states on target network
        target_q = self.target_model(next_states)
        # create array [#states x #actions]: for selected action write target_q value, for other actions write 0
        target_value = tf.reduce_sum(tf.one_hot(next_action, self.num_act) * target_q, axis=1)

        # Q-values of successor states plus reward for current transition
        # if done=1 -> no successor -> consider current reward only
        target_value_update = (1 - dones) * self.gamma * target_value + rewards
        target_value_orig = self.model(states)

        # update target_value_orig with target_value_update on positions of chosen actions
        target_value_ges = tf.tensor_scatter_nd_update(target_value_orig, tf.stack([idx, actions], axis=1),
                                                       target_value_update)

        # get all trainable variables of the model
        dqn_variable = self.model.trainable_variables

        with tf.GradientTape() as tape:
            # forward pass for states
            logits = self.model(states)
            # compute loss
            loss = self.loss(tf.stop_gradient(target_value_ges), logits)

        # compute gradient
        dqn_grads = tape.gradient(loss, dqn_variable)

        # apply gradient
        self.opt.apply_gradients(zip(dqn_grads, dqn_variable))

    def save_model(self): #function to save the current neural net and its weights to a file, 
        #to be able to load it later and to avoid training the model again
        
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - save the current neural net and its weights to a file
        #     Hint: use *.json for model structure and *.h5 (if you use Keras 2) / *.weights.h5 (if you use Keras 3) for weights

        ### Enter your code here ###
        model_json = self.model.to_json()
        with open(os.path.join(script_dir, "dqn_model.json"), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(script_dir, "dqn_model.weights.h5"))
        ### End of your code ###

    def load_model(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - load a neural net and its weights from a file into the model variable of class Network
        #   - make sure to handle exceptions, if no model was saved (e.g. FLAG_PLAY_ONLY=True without previous training)

        ### Enter your code here ###
        try:
            with open(os.path.join(script_dir, "dqn_model.json"), "r") as json_file:
                loaded_model_json = json_file.read()
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(os.path.join(script_dir, "dqn_model.weights.h5"))
        except Exception as e:
            print("No model found. Train the model first.")
        ### End of your code ###


class Agent: #class for the agent, who plays the game and trains the neural network
    def __init__(self, max_episodes=MAX_TRAINING_EPISODES):
        # Input:
        #   - self
        #   - max_episodes: number of maximum episodes for training
        # Return:
        #   - none
        # Function:
        #   - create a environment
        #   - store max_episodes
        #   - create a new class variable epsilon to store the current exploration rate and set it to 1
        #   - create 2 class variables for number of possible actions and observations
        #   - create a net

        ### Enter your code here ###
        self.env = gym.make("LunarLander-v2") #, render_mode="human")
        self.max_episodes = max_episodes
        self.epsilon = 1.0
        self.num_actions = self.env.action_space.n
        self.num_observations = self.env.observation_space.shape[0]
        self.net = Network(self.num_observations, self.num_actions) #creating a new neural network instead of q-tabke which is used for training
        ### End of your code ###

    def update_epsilon(self, episode):
        # Input:
        #   - self
        #   - episode: number of current episode
        # Return:
        #   - none
        # Function:
        #   - update the class variable epsilon using epsilon = DECAY_RATE ^ episode
        #   - make sure, that epsilon won't fall below MIN_EXP_RATE

        ### Enter your code here ###

        self.epsilon = max(EXP_RATE_MIN, DECAY_RATE ** episode)

        ### End of your code ###

    def choose_action(self, state):
        # Input:
        #   - self
        #   - state: current state of the environment
        # Return:
        #   - action: integer which action was chosen
        # Function:
        #   - compute an action based on the input state and epsilon-greedy strategy

        ### Enter your code here ###

        if np.random.rand() < self.epsilon: #exploration. He chooses a random action compared to the exploitation where he chooses the action with the highest Q-value
            return self.env.action_space.sample()
        q_values = self.net.model.predict(np.array([state]), verbose=0) #exploitation
        return np.argmax(q_values[0])

        ### End of your code ###

    def train(self): #function to train the agent to play the game and save the model to a file to be able 
        #to load it later instead of training the model again
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - initialize LearningStats object
        #   - train for TRAIN_EPISODES or until mean reward over the last MEAN_REWARD_EPISODES > MEAN_SUCCESS
        #   - run each episode until done==True
        #   - for each episode update epsilon for epsilon-greedy strategy
        #   - after each step update the neural net
        #     (only if enough samples in ReplayBuffer to fill a complete Mini Batch)
        #   - after each step add the result of the current episode to the LearningStats object
        #   - for each step print the current statistics:
        # Ep 320  Time (total): 0 h 4 min 49 s  Steps: 84  Time (episode): 0.92  Reward: -547.24  mean Reward: -492.48
        #     Use convert_time() to convert seconds to h, min, s
        #   - when the training is done, plot the statistics of the training process and save the model using save_model() function

        ### Enter your code here ###

        stats = LearningStats()
        start_time = time.time()
        for episode in range(self.max_episodes):
            state = self.env.reset()[0]
            done = False
            total_reward = 0
            steps = 0
            while not done and steps < MAX_TRAINING_STEPS: #run each episode until done==True, but not more than MAX_TRAINING_STEPS
                action = self.choose_action(state)#choose an action based on the current state
                next_state, reward, done, _, _ = self.env.step(action) #done: True if the episode is over, False otherwise
                self.net.replay_buffer.append((state, action, next_state, reward, done)) #buffer: store the last 500000 transitions of the agent
                state = next_state
                total_reward += reward
                steps += 1
                self.net.prepare_update()
            stats.append_step(total_reward, steps)
            self.update_epsilon(episode) #update epsilon for epsilon-greedy strategy
            elapsed_time = time.time() - start_time
            h, m, s = convert_time(elapsed_time) #for converting time in seconds to hours, minutes and seconds, to print the time in a readable format
            print(f"Ep {episode} Time: {h} h {m} min {s} s Steps: {steps} Reward: {total_reward:.2f} Mean Reward: {stats.mean_rewards[-1]:.2f}")
            if stats.mean_rewards[-1] >= MEAN_SUCCESS:
                break
        stats.plot_stats()
        self.net.save_model() #save the model to a file to be able to load it later instead of training the model again

        ### End of your code ###

    def play(self): #function to play the game and render(visualise) one episode and print the total reward
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - if PLAG_PLAY_ONLY==TRUE, load model
        #   - modify self.epsilon in order to convert epsilon-greedy strategy to best-action strategy
        #   - play and render one episode
        #   - print the achieved reward

        ### Enter your code here ###
        self.env = gym.make("LunarLander-v2", render_mode="human")

        if FLAG_PLAY_ONLY:
            self.net.load_model()
        self.epsilon = 0.0
        state = self.env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            self.env.render()
            action = self.choose_action(state)
            state, reward, done, _, _ = self.env.step(action)
            total_reward += reward
        self.env.close()
        print(f"Total Reward: {total_reward:.2f}")

        ### End of your code ###


if __name__ == "__main__":
    agent = Agent()

    # don't train, if FLAG_PLAY_ONLY is activated
    if not FLAG_PLAY_ONLY:
        agent.train()
    agent.play()