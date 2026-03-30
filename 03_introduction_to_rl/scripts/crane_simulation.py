import numpy as np
import matplotlib.pyplot as plt
import gym 

# Source: Chapter Machine Learning, Lecture ISMLP, AIS 2021
#Step 1: Define the grid and all required variables in the CraneSim class
class CraneSim: 
    ## Initialize starting data
    def __init__(self):

        # Input:
        #   - self
        # Return:
        #   - none
        # Function:
        #   - initialize all required parameters for the crane simulation such as the grid, position of obstacles, ...

        #   - 0 1 2 3 4 5 6 7 8 9
        #   0       
        #   1       
        #   2     O           O
        #   3     O           O
        #   4     O           O
        #   5     O           O
        #   6     O           O
        #   7     O           O
        #   8     O           O
        #   9 S               O T
        # S: Start, O: Obstacle, T: Target
        
        # Size of the problem
        self.height =10
        self.width =10
        # self.grid [np-array, dimensions: self.height x self.width] stores the reward the agent earns when going to a specific position
        # start by initializing the reward for all positions with -1
        self.grid =np.full((self.height, self.width), -1)  #full: initialing the matrix with values of -1

        # define start position: tuple; format: (height, width)
        self.current_location =(9, 0)

        # Obstacles: Save all obstacle positions in a list of tuples
        self.obstacle_list =[(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2),
                              (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8)]
        
        # Save the Target Location: tuple
        self.target_location = (9, 9)

        # All terminal states: list of tuples; when reaching those states, the simulation ends
        self.terminal_states = self.obstacle_list + [self.target_location]
        
        # Rewards: set the reward stored in self.grid for all obstacles to -100 and for the target position to +100
        for obstacle in self.obstacle_list:
            self.grid[obstacle[0], obstacle[1]] = -100 #row, column
        self.grid[self.target_location[0], self.target_location[1]] = 100
        # Define possible actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']

        self.history = []  # Save the history of the agent's positions

    def get_available_actions(self):
        # Returns available actions
        return self.actions
    
    def agent_on_map(self):
    # Returns grid with current position
        grid = np.zeros((self.height, self.width))  # 1️⃣ Create an empty map
        grid[self.current_location[0], self.current_location[1]] = 1  # 2️⃣ Mark the agent's position
        return grid  # 3️⃣ Return the grid

    
    def get_reward(self, new_location):
        # Returns the reward for an input position
        return self.grid[new_location[0], new_location[1]] 
    #📌 If new_location = (1,1) (an obstacle), the function returns -100.
    #📌 If new_location = (1,3) (the target), the function returns 100.
    #📌 If new_location = (0,0), the function returns -1 (default penalty).
    
# for the drow the results of the positions of the agent
    def print_current_position(self):
        print("")
        string = "-"
        for y in range(environment.width):
            string += "-" + str(y) + "-"
        print(string) 

        for x in range(environment.height):
            string=str(x)
            for y in range(environment.width):
                ende = False
                for i in range(len(self.obstacle_list)):
                    obstacle = self.obstacle_list[i]
                    if obstacle[0] == x and obstacle[1] == y:
                        string+="-H-"
                        ende = True

                if ende == False:
                    if self.current_location[0] == x and self.current_location[1] == y:
                        string+="-X-"
                        ende = True
                    elif self.target_location[0] == x and self.target_location[1] == y:
                        string+="-G-"
                        ende = True

                if ende == False:
                    for i in range(len(self.history)):
                        step = self.history[i]
                        if step[0] == x and step[1] == y:
                            string+="-x-"
                            ende = True
                            break

                if ende == False:
                    string+="---"

            print(string)
    
    #Example def make_step(self, action):
    #✅ Why Is This Useful?
    #Visual debugging for reinforcement learning environments.
    #Helps understand the agent's position and movement.
    #Makes the simulation more interactive.
    #--0--1--2--3--4--5--6--7--8--9-
#0------------------------------
#1------------------------------
#2-------H--------------H-------
#3-------H--------------H-------
#4-------H--------------H-------
#5-------H--------------H-------
#6-------H--------------H-------
#7-------H--------------H-------
#8-------H--------------H-------
#9-X--------------------------G-
    def make_step(self, action):
        # Moves agent; if agent is tries to go over boundary, he doesn't move, but gets a negative reward
        # The function returns the reward for a step

        # Save last location
        last_location = self.current_location
        
        # UP
        if action == 'UP':
            # If agent is on upper boundary, the position doesn't change
            if last_location[0] == 0:
                reward = self.get_reward(last_location)
            # Else: Move upwards = increase y-component
            else:
                self.current_location = ( self.current_location[0] - 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
        
        # DOWN
        elif action == 'DOWN':
            # If agent is on lower boundary, the position doesn't change
            if last_location[0] == self.height - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0] + 1, self.current_location[1])
                reward = self.get_reward(self.current_location)
            
        # LEFT
        elif action == 'LEFT':
            # If agent is on left boundary, the position doesn't change
            if last_location[1] == 0:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] - 1)
                reward = self.get_reward(self.current_location)

        # RIGHT
        elif action == 'RIGHT':
            # If agent is on right boundary, the position doesn't change
            if last_location[1] == self.width - 1:
                reward = self.get_reward(last_location)
            else:
                self.current_location = ( self.current_location[0], self.current_location[1] + 1)
                reward = self.get_reward(self.current_location)
                
        # Save history
        self.history.append(self.current_location)

        return reward
    
    def check_state(self):
        # Check if agent is in terminal state (=target or obstacle)
        if self.current_location in self.terminal_states:
            return 'TERMINAL'


#Step 2: Implement the epsilon-greedy policy and the Q-Table update function in the Q_Agent class
class Q_Agent():
    # Initialize
    def __init__(self, environment, epsilon=0.05, alpha=0.1, gamma=0.9):
        self.environment = environment
        self.q_table = dict() # all Q-values in dictionary
        for x in range(environment.height): 
            for y in range(environment.width):
                self.q_table[(x,y)] = {'UP':0, 'DOWN':0, 'LEFT':0, 'RIGHT':0} # Initialize all values with 0

        self.epsilon = epsilon  # Set the exploration rate (epsilon) for the epsilon-greedy policy.
                        # Epsilon determines the probability of choosing a random action instead of the best-known action.
                        # A higher epsilon encourages more exploration, while a lower epsilon favors exploitation.

        self.alpha = alpha      # Set the learning rate (alpha) for updating the Q-values.
                        # The learning rate controls how much new information overrides the old Q-value.
                        # A higher alpha means faster learning, but too high can lead to instability.

        self.gamma = gamma      # Set the discount factor (gamma) for future rewards.
                        # Gamma determines the importance of future rewards compared to immediate rewards.
                        # A higher gamma (closer to 1) makes the agent prioritize long-term rewards,
                        # while a lower gamma (closer to 0) makes the agent focus on short-term rewards.
    
    def choose_action(self, available_actions,no_random=False):
        # Input:
        #   - self
        #   - available_actions: array of available actions ['UP', 'DOWN', 'LEFT', 'RIGHT']
        #   - no_random: bool
        # Return:
        #   - action: String; has to be one of the available actions out of available_actions
        # Function:
        #   - if no_random is set to true, always return the best action, otherwise use epsilon-greedy strategy as described in slides

        ### Enter your code here ###
        if no_random or np.random.rand() > self.epsilon:
            # Choose the best action
            state = self.environment.current_location
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            actions_with_max_q = [action for action, q in q_values.items() if q == max_q]
            return np.random.choice(actions_with_max_q)
        else:
        # Choose a random action
            return np.random.choice(available_actions)

        ### End of your code ###
    
    def update(self, old_state, reward, new_state, action):
        # Input:
        #   - self
        #   - old_state: position in grid before making a step
        #   - reward: earned reward for executed action
        #   - new state: new position in the grid based on old_state and action
        #   - action: chosen action
        # Return:
        #   - none
        # Function:
        #   - Update the Q-Table using the Q-Learning formula

        ### Enter your code here ###
        old_q_value = self.q_table[old_state][action]
        max_future_q = max(self.q_table[new_state].values())
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q - old_q_value)
        self.q_table[old_state][action] = new_q_value

        ### End of your code ###


def play(environment, agent, trials=500, max_steps_per_episode=1000, learn=False, eval = False):
    # Input:
    #   - environment: variable of class CraneSim
    #   - agent: variable of class Q_Agent
    #   - trials: number how many episodes (=games) the Agent will play during the training-process
    #   - max_steps_per_episodes: maximum steps per episode/game/trial
    #   - learn: Bool; shows if Q-Table should be updated
    #   - eval: Bool; shows if Q-Table should only be evaluated
    # Return:
    #   - reward_per_episode: array[number of episodes]; saves the reward earned in each episode
    # Function:
    #   - Run as many trials as specified
    #   - Perform Q-Learning after each action
    #   - If maximum number of steps or a terminal state is reached, save the reward of the current episode, reset the environment and start a new trial
    #   - Distinguish for different states of bool-variables learn and eval
    #   - If eval==True print out the positions of the agent


    # this function iterates and updates the Q-Table, if necessary
    reward_per_episode = [] # Initialize performance log
    
    #if eval == True:
    #    environment.print_current_position()

    for trial in range(trials): # Run trials
        cumulative_reward = 0 # Initialize values of each game
        step = 0
        game_over = False

        ### Enter your code here ###
        while not game_over and step < max_steps_per_episode:
            old_state = environment.current_location
            action = agent.choose_action(environment.get_available_actions(), no_random=eval)
            reward = environment.make_step(action)
            new_state = environment.current_location
            
            cumulative_reward += reward
            
            if learn:
                agent.update(old_state, reward, new_state, action)
            
            if environment.check_state() == 'TERMINAL':
                game_over = True
            
            step += 1

        reward_per_episode.append(cumulative_reward)
        
        if eval:
            environment.print_current_position()
        
        # Reset environment for the next trial
        environment.current_location = (9, 0)
        environment.history = []


        ### End of your code ###

    # Return performance log
    return reward_per_episode

#Step 4: Initialize the environment and agent, train the agent, and evaluate it
# Initialize environment and agent
environment = CraneSim()
#agentQ = Q_Agent(environment, epsilon=0.05, alpha=0.1, gamma=0.9)
#Tune the hyperparameter: epsilon, alpha, and gamma
agentQ = Q_Agent(environment, epsilon=0.5, alpha=0.95, gamma=0.95) 
#Decrease epsilon:Goal: Reduce the probability of random actions (exploration) and increase the likelihood
# of choosing the best-known action (exploitation).
#Increase alpha:Goal: Speed up learning by giving more weight to new information.
#Decrease gamma:Goal: Focus on short-term rewards and reduce the importance of future rewards.

## Train agent
reward_per_episode = play(environment, agentQ, trials=500, learn=True, eval = False)

# Simple learning curve
plt.plot(reward_per_episode)
plt.show()

# Evaluate agent
reward_per_episode = play(environment, agentQ, trials=1, learn=False, eval = True)

#Result interpretation:
#Reinforcement Learning Training (Reward per Episode)
#The graph could show rewards per episode. The early episodes have large negative values
# (agent is learning), 
# and later episodes stabilize, suggesting improved performance.
