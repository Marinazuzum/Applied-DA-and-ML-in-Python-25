import numpy as np
import random
import gym
from tqdm import tqdm
import matplotlib.pyplot as plt

#Есть среда (игра) 🎮
#Например, CartPole – это игра, где тележка должна балансировать шест, чтобы он не упал.

#Есть агент (умный бот) 🤖
#Это программа, которая учится играть в эту игру.
#Агент делает действия 🎯
#Например, двигает тележку влево или вправо.
#Среда дает награду 🏆
#Если шест стоит прямо → +1 очко
#Если упал → игра закончена
#Агент учится 📈
#Он пробует разные стратегии и запоминает, какие действия приносят больше очков.


# These parameters are only a suggestion. Feel free to change them in order to achieve better and more stable results
#📌Why these changes?
#📌Slower exploration decay allows the agent to explore more before settling into an exploitation strategy.
#📌Reduced training episodes and bucket sizes help make the learning process more efficient.
#📌Slight adjustments in the learning rate and success threshold should help the agent reach the desired level of performance faster.

# Number of buckets for each observation
#BUCKETS = (3, 3, 12, 24)
BUCKETS = (3, 3, 8, 16)
#Try experimenting with smaller bucket sizes, earning without losing too much precision
# Minimum value for the exploration rate
MIN_EXP_RATE = 0.025
# decay rate for the exploration rate
DECAY_RATE = 0.99 #exponrntial decrease
#DECAY_RATE = 0.97 #exponrntial decrease quicker
#Эпизод	epsilon = 0.99^episode
#1	 0.99
#10	 0.904
#50	 0.605
#100 0.366
# Maximum number of training episodes
#MAX_TRAIN_EPISODES = 100000 #stabilize after 4000 episodes.
MAX_TRAIN_EPISODES = 2000
# Number of steps after which a single episode is terminated
MAX_STEPS = 500
#MAX_STEPS = 700
#e.g., 700–1000 steps) might give the agent more time to explore the environment
# Learning rate used for updating the Q-Table
LEARNING_RATE = 0.01
#If you notice that the agent is converging too slowly, increasing the learning
# rate slightly to 0.02 might help speed up the learning process

# Discount factor used  for updating the Q-Table
GAMMA = 1
# Necessary mean success to consider task as solved
#MEAN_SUCCESS = 200
MEAN_SUCCESS = 150
#You might want to try 150 instead of 200, especially if your agent is taking longer to converge.
# Number of episodes regarded to compute the mean reward
MEAN_REWARD_EPISODES = 50  #moving average

# Remark: Increase MEAN_SUCCESS (e.g. to 300) and decrease MAX_STEPS (e.g. to 400) to watch an agent pole with high
#         probability to balance the pole for a very long time


class LearningStats:
    def __init__(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - define empty list for rewards and mean_rewards

        ### Enter your code here ###
        self.rewards = []
        self.mean_rewards = []


        ### End of your code ###

    def append_step(self, reward):
        # Input:
        #   - self
        #   - reward: reward of episode
        # Return:
        #   - none
        # Function:
        #   - Compute mean reward for last MEAN_REWARD_EPISODES episodes
        #   - If less than MEAN_REWARD_EPISODES have been trained so far, set the mean reward to 0
        #   - Append reward and mean reward to corresponding list

        ### Enter your code here ###
        self.rewards.append(reward)
        mean_reward = np.mean(self.rewards[-MEAN_REWARD_EPISODES:]) if len(self.rewards) >= MEAN_REWARD_EPISODES else 0 #if we have less than 50 episodes, we set the mean reward to 0
        self.mean_rewards.append(mean_reward)



        ### End of your code ###

    def plot_stats(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - Plot reward per episode, mean reward and a horizontal line to mark the necessary mean reward,
        #     which has to be reached to consider the task as solved
        #   - Make sure, that the mean reward and the horizontal line are not covered by the line of reward per episode
        #     Hint: See task sheet for example and use 'zorder'
        #   - Add labels to the axes and a legend

        ### Enter your code here ###
        plt.figure(figsize=(12, 5))
        plt.plot(self.rewards, label='Reward per Episode', alpha=0.5)
        plt.plot(self.mean_rewards, label='Mean Reward', color='r', linewidth=2)
        plt.axhline(y=MEAN_SUCCESS, color='g', linestyle='--', label='Success Threshold')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()

        ### End of your code ###


class Environment:
    def __init__(self, buckets=BUCKETS):
        # Input:
        #   - self
        #   - buckets: number of discrete intervals for each of the four observations
        #     (cart position, cart velocity, pole angle, pole velocity)
        # Return:
        #   - none
        # Function:
        #   - create a new cart pole gym environment
        #   - save the number of buckets for later use

        ### Enter your code here ###
        self.env = gym.make('CartPole-v1', render_mode="human")
        self.buckets = buckets
        self.obs_limits = list(zip(self.env.observation_space.low, self.env.observation_space.high))
        self.obs_limits[1] = (-2, 2)
        self.obs_limits[3] = (-2, 2)

        ### End of your code ###

    def discretize(self, obs):
        # Input:
        #   - self
        #   - obs: current state of environment
        # Return:
        #   - new_obs: tuple, in which a new observation is stored
        # Function:
        #   - discretize the continuous observation for cart position, cart velocity, pole angle and pole velocity
        #   - get the limits of each observation of the environment
        #   - think about more realistic bounds for cart and pole velocity, which are necessary to compute the buckets
        #     (the speed limits are very high. If the cart or pole moves with that speed, you have definitely lost the
        #     game already. So we are not interested in most of the range of possible velocities.)
        #   - use the specified lower and upper bound for each observation to assign a bucket to each input observation
        #   - observations outside the specified bounds should be put to the lowest or highest bucket

        ### Enter your code here ###
        ratios = [(obs[i] - self.obs_limits[i][0]) / (self.obs_limits[i][1] - self.obs_limits[i][0]) for i in range(len(obs))]
        new_obs = [int(min(max(r * self.buckets[i], 0), self.buckets[i] - 1)) for i, r in enumerate(ratios)]
        ### End of your code ###

        return tuple(new_obs)

#The agent is the learner in the reinforcement learning process.
#It interacts with the environment, receives rewards, and learns to take actions that maximize the rewards.
#The agent uses a Q-table to store the expected rewards for each state-action pair.
#During training, the agent updates the Q-values based on the rewards received and the expected future rewards.
#The agent uses an epsilon-greedy strategy to balance exploration and exploitation.
#During training, the agent gradually reduces the exploration rate (epsilon) to focus more on exploiting the learned Q-values.
#After training, the agent uses a greedy strategy to choose actions based on the learned Q-values.
class Agent:
    def __init__(self, lr=LEARNING_RATE, max_episodes=MAX_TRAIN_EPISODES):
        # Input:
        #   - self
        #   - lr: Learning rate
        #   - max_episodes: number of maximum episodes for training
        # Return:
        #   - none
        # Function:
        #   - create variable of class environment
        #   - store all input variables as class variables
        #   - create a new class variable epsilon to store the current exploration rate and set it to 1
        #   - create a Q-table

        ### Enter your code here ###
        self.env = Environment()
        self.lr = lr
        self.max_episodes = max_episodes
        self.epsilon = 1.0
        self.q_table = np.zeros(self.env.buckets + (self.env.env.action_space.n,))

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
        self.epsilon = max(MIN_EXP_RATE, DECAY_RATE ** episode)#decay rate is 0.99, and the minimum exploration rate is 0.025
        # so, we are reducing the exploration rate by 1% after each episode, and the minimum exploration rate is 0.025


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
        if random.uniform(0, 1) < self.epsilon: #uniform distribution, between 0 and 1, we have 1 so 1 episode TRUE
            return self.env.env.action_space.sample() #exloraition
        else:
            return np.argmax(self.q_table[state]) #exploitation
        ### End of your code ###

    def update_q_table(self, state, action, reward, new_state):
        # Input:
        #   - self
        #   - state: current state of the environment
        #   - action: chosen action
        #   - reward: reward earned based on the state and action
        #   - new_state: new state of environment based on stae and action
        # Return:
        #   - none
        # Function:
        #   - update value of Q-table for state action tuple

        ### Enter your code here ###
        best_future_q = np.max(self.q_table[new_state])
        self.q_table[state + (action,)] += self.lr * (reward + GAMMA * best_future_q - self.q_table[state + (action,)])


        ### End of your code ###

    def train(self): #train the agent, and depending on the mean reward over the last MEAN_REWARD_EPISODES, we can stop the training
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - initialize LearningStats object
        #   - train for TRAIN_EPISODES or until mean reward over the last MEAN_REWARD_EPISODES > MEAN_SUCCESS
        #   - run each episode until done==True or the number of MAX_STEPS is reached
        #   - for each episode update epsilon for epsilon-greedy strategy
        #   - after each step update the Q-Table
        #   - after each step add the result of the current episode to the LearningStats object
        #   - when the training is done, plot the statistics of the training process

        ### Enter your code here ###
        stats = LearningStats()
        for episode in tqdm(range(self.max_episodes)):
            state = self.env.discretize(self.env.env.reset()[0])
            total_reward = 0
            for _ in range(MAX_STEPS):
                action = self.choose_action(state) #choose action
                obs, reward, done, _, _ = self.env.env.step(action)
                new_state = self.env.discretize(obs)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                total_reward += reward
                if done:
                    break
            stats.append_step(total_reward)
            self.update_epsilon(episode) #update epsilon after each episode, so with each episode, the agent becomes more greedy (epsilon decreases with decay rate)
            if np.mean(stats.rewards[-MEAN_REWARD_EPISODES:]) >= MEAN_SUCCESS:
                print("Training successful!")
                break
        stats.plot_stats()


        ### End of your code ###

    def play(self):
        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - modify self.epsilon in order to convert epsilon-greedy strategy to best-action strategy
        #   - play and render one episode
        #   - print the achieved reward

        ### Enter your code here ###
        #when we play, we want to exploit the learned Q-values
        self.epsilon = 0 #so, that is why we set epsilon to 0 (greedy strategy)
        state = self.env.discretize(self.env.env.reset()[0])
        total_reward = 0
        for _ in range(MAX_STEPS):
            self.env.env.render()  # Render the environment
            action = self.choose_action(state)
            obs, reward, done, _, _ = self.env.env.step(action)
            new_state = self.env.discretize(obs)
            state = new_state
            total_reward += reward
            if done:
                break
        print(f"Total Reward: {total_reward}")
        self.env.env.close()
        ### End of your code ###

#if the test results with reward >200 can be seen as success.
#Max Reward at Play time should be 500 now(predifined in environment CartPole).

if __name__ == "__main__":
    agent = Agent()
    agent.train()
    agent.play() 

#Interpretation of the Graph:
#X-axis (Episodes) – Number of training episodes.
#Y-axis (Reward) – Reward obtained by the agent per episode.
#Light blue area – Variation in rewards across episodes.
#Red line – Moving average of rewards (Mean Reward).
#Green dashed line – Success threshold (Success Threshold).

#Key Insights:
#At the beginning, the agent receives low rewards.
#For the first 4000 episodes, the average reward remains low, indicating slow learning progress.
#After episode 4000, there is a significant increase in the mean reward, suggesting that the agent has learned a better strategy.
#By episode 7000, the mean reward (red line) surpasses the success threshold (green line), meaning the agent has successfully learned to perform the task.
#This graph indicates successful reinforcement learning, where the model gradually improves its strategy and reaches the desired performance level.