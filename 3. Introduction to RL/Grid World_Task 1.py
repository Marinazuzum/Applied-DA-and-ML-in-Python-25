
#Grid World – Task 1:
#Small Grid World: 2x2
#- Go from a starting point (S) to a target location (T)
#- Possible actions: up, down, left or right (crashing into the boundary → position remains the same, but
#counts as new step for reward calculation)
#- Rewards:
#𝒙𝒕+𝟏 𝒓𝒕+𝟏(𝒙𝒕+𝟏)
#E -1
#T +100
#B -0.5
#S -1

#Task 1:
#• Perform the first iteration of Q-Learning (Steps 1-5) by hand
#• Use learning rate 𝛼 = 0.1 and discount factor 𝛾 = 1
#• Neglect epsilon-greedy strategy
# The agent does not explore new actions and always chooses the best-known action
#1. Initialize Q-Table
#2. Choose an action:
#• Go to current state (S)
#• All actions have the same Q-value → choose randomly: “right“
#3. Perform an action: 𝑥𝑡+1 = (B)
#4. Measure reward: r = -0.5
#5. Update Q-Table:
#• Go to new state: (B) → highest Q-value: 0
#• Compute updated Q-Value for (S | right)

import numpy as np

# Define the grid world 
# Dictionury, Possible actions and the resulting next state.
#Since all Q-values are initially 0, the action is chosen randomly.
grid = {
    'S': {'reward': -1, 'actions': {'up': 'E', 'down': 'S', 'left': 'S', 'right': 'B'}},
    'E': {'reward': -1, 'actions': {'up': 'E', 'down': 'S', 'left': 'E', 'right': 'E'}},
    'B': {'reward': -0.5, 'actions': {'up': 'B', 'down': 'T', 'left': 'S', 'right': 'B'}},
    'T': {'reward': 100, 'actions': {'up': 'B', 'down': 'T', 'left': 'T', 'right': 'T'}}
}

# Initialize Q-Table
q_table = {
    'E': {'up': 0, 'down': 0, 'left': 0, 'right': 0},
    'T': {'up': 0, 'down': 0, 'left': 0, 'right': 0},
    'S': {'up': 0, 'down': 0, 'left': 0, 'right': 0},
    'B': {'up': 0, 'down': 0, 'left': 0, 'right': 0}
}

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 1.0  # Discount factor

# Step 1: Start at state S
current_state = 'S'

# Step 2: Choose an action (right, as per example)
action = 'right'

# Step 3: Perform the action and observe the next state
next_state = 'B'  # From S, action 'right' leads to B

# Step 4: Measure the reward
reward = -0.5  # Reward for state B

# Step 5: Update the Q-Table
max_future_q = max(q_table[next_state].values())  # Maximum Q-value for the next state (B)
current_q = q_table[current_state][action]  # Current Q-value for the chosen action (right)

# Q-Learning formula
new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
q_table[current_state][action] = new_q
#Q(S,right)=(1−0.1)⋅0+0.1⋅(−0.5+1⋅0)=−0.05

# Print the updated Q-Table
print("Updated Q-Table:")
for state, actions in q_table.items():
    print(f"{state}: {actions}")


#Task 2:
#• After 5 completed episodes with epsilon-greedy strategy (𝜖 = 0.05) the Q-Table looks like shown
#below
#• Compute the first iteration of Q-Learning for episode 6, starting again at (S) by hand
#• As before: Neglect epsilon-greedy strategy