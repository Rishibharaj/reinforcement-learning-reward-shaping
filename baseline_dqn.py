import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

# Create the LunarLander-v2 environment
env = gym.make("LunarLander-v2")

# Get the shape of the state and number of possible actions
state_shape = env.observation_space.shape[0]  # 8-dimensional vector
num_actions = env.action_space.n              # 4 discrete actions

# Hyperparameters
learning_rate = 0.001
gamma = 0.99                  # Discount factor for future rewards
epsilon = 1.0                 # Initial exploration rate
epsilon_decay = 0.995         # Decay rate for epsilon
epsilon_min = 0.01            # Minimum exploration rate
batch_size = 64
memory_size = 100000
num_batches = 2000
episodes_per_batch = 64

# Experience replay buffer
memory = deque(maxlen=memory_size)

# Build a simple feedforward Q-network
def build_q_network():
    model = tf.keras.Sequential([
        layers.Input(shape=(state_shape,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions, activation='linear')  # Output Q-values for each action
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Initialize the Q-network and target network
q_network = build_q_network()
target_network = build_q_network()
target_network.set_weights(q_network.get_weights())  # Sync weights initially

# Epsilon-greedy action selection
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    q_values = q_network.predict(np.expand_dims(state, axis=0), verbose=0)
    return np.argmax(q_values[0])        # Exploit

# Train the Q-network using a mini-batch from memory
def train_q_network():
    if len(memory) < batch_size:
        return  # Not enough samples to train
    # Sample a random batch of experiences
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)
    states = np.array(states)
    next_states = np.array(next_states)
    # Predict Q-values for current and next states
    q_values = q_network.predict(states, verbose=0)
    q_next = target_network.predict(next_states, verbose=0)
    # Update Q-values using the Bellman equation
    for i in range(batch_size):
        if dones[i]:
            q_values[i][actions[i]] = rewards[i]
        else:
            q_values[i][actions[i]] = rewards[i] + gamma * np.max(q_next[i])
    # Train the network on the updated Q-values
    q_network.fit(states, q_values, epochs=1, verbose=0)

# Main training loop
for batch in range(num_batches):
    total_rewards = []
    for episode in range(episodes_per_batch):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            # Store the experience in memory
            memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            # Train the network after each step
            train_q_network()
        total_rewards.append(episode_reward)
    # Decay epsilon after each batch
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    # Update the target network
    target_network.set_weights(q_network.get_weights())
    # Print average reward for this batch
    print(f"Batch {batch + 1}/{num_batches}, Average Reward: {np.mean(total_rewards):.2f}")

# Close the environment
env.close()
