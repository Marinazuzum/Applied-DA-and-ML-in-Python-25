import numpy as np
import random

# Define the grid world dimensions
grid_size = (2, 2)

# Define the rewards for each state (row, col)
rewards = {
    'S': -1,  # Start
    'E': -1,  # Empty
    'T': 100, # Target
    #'B': -0.5 # Blocked
    'B': +1 # Blocked (Task 4)
}

# State representation (row, col) for S, E, T, B
states = {
    'S': (1, 0),  # Start at (1, 0)
    'E': (0, 0),  # Empty at (0, 0)
    'T': (0, 1),  # Target at (0, 1)
    'B': (1, 1),  # Blocked at (1, 1)
}

# Actions: up, down, left, right
actions = ['up', 'down', 'left', 'right']

# Define Q-table (rows = states, columns = actions)
Q_table = np.zeros((4, 4))  # 4 states and 4 actions

# Mapping for actions to movement (row, col) changes
action_map = {
    'up': (-1, 0),  # move up
    'down': (1, 0), # move down
    'left': (0, -1), # move left
    'right': (0, 1), # move right
}

# Helper function to get the state index (0 = S, 1 = E, 2 = T, 3 = B)
def get_state_index(state):
    state_index_map = {'S': 0, 'E': 1, 'T': 2, 'B': 3}
    return state_index_map[state]

# Initialize parameters
alpha = 0.1  # Learning rate
gamma = 1.0  # Discount factor
epsilon = 0.9999  # Exploration rate (starts high for exploration)
epsilon_min = 0.1 # Minimum exploration rate
epsilon_decay = 1.0 # Decay rate for epsilon
num_episodes = 1000 # Number of episodes

# Function to get next state based on action
def get_next_state(state, action):
    (row, col) = states[state]
    (action_row, action_col) = action_map[action]
    new_row = max(0, min(row + action_row, grid_size[0] - 1))
    new_col = max(0, min(col + action_col, grid_size[1] - 1))
    
    # Map the new position back to the state
    for key, (r, c) in states.items():
        if (r, c) == (new_row, new_col):
            return key
    return state  # Return the same state if the move is invalid (e.g., out of bounds)

# Function to update Q-table
def update_Q(state, action, reward, next_state):
    state_idx = get_state_index(state)
    action_idx = actions.index(action)
    next_state_idx = get_state_index(next_state)
    
    # Q-learning update rule
    Q_table[state_idx, action_idx] += alpha * (
        reward + gamma * np.max(Q_table[next_state_idx, :]) - Q_table[state_idx, action_idx]
    )

# Training loop
for episode in range(num_episodes):
    state = 'S'  # Start from state S
    done = False
    while not done:
        # Choose action based on epsilon-greedy strategy
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)  # Exploration: random action
        else:
            action = actions[np.argmax(Q_table[get_state_index(state), :])]  # Exploitation: best action
        
        # Get the next state based on action
        next_state = get_next_state(state, action)
        
        # Get the reward for the next state
        reward = rewards[next_state]
        
        # Update the Q-table
        update_Q(state, action, reward, next_state)
        
        # If we reach the target (T), stop the episode
        if next_state == 'T':
            done = True
        
        # Move to the next state
        state = next_state
    
    # Decay epsilon for exploration-exploitation balance
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Print the final Q-table
print("Final Q-table:")
print(Q_table)


#####
###− Task 4: What happens, if we apply the following reward structure to the Grid World problem?
#𝒙𝒕+𝟏 𝒓(𝒙𝒕+𝟏)
#E   -1
#T   +100
#B   +1
#S   -1

#The algorithm will learn, that the highest reward can be achieved, if it steps on (B) as often as possible
#• For 𝑥𝑡 = (B) the preferred actions are will be “down“ and “right“ so that 𝑥𝑡+1 = (B) and 𝑟𝑡+1 = +1
#• The algorithm will never stop in order to maximize its reward
#→ Always be careful, when defining the reward function!
