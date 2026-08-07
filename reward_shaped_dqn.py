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
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
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
        layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Initialize the Q-network and target network
q_network = build_q_network()
target_network = build_q_network()
target_network.set_weights(q_network.get_weights())

# Epsilon-greedy action selection
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    q_values = q_network.predict(np.expand_dims(state, axis=0), verbose=0)
    return np.argmax(q_values[0])

### Reward Shaping Function
# ============================================================
# Reward Shaping
#
# R' = Renv - 0.1|x| - 0.1|vy| - 0.1|theta|
#
# where:
#   x      = horizontal position
#   vy     = vertical velocity
#   theta  = lander angle
#
# This provides additional learning signals intended to
# encourage centered positioning, controlled descent,
# and stable orientation.
# ============================================================
```python
def shape_reward(state, reward):
    x = state[0]
    vy = state[3]
    angle = state[4]

    shaped = reward
    shaped += -0.1 * abs(x)
    shaped += -0.1 * abs(vy)
    shaped += -0.1 * abs(angle)

    return shaped
```

# Train the Q-network using a mini-batch from memory
def train_q_network():
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)
    states = np.array(states)
    next_states = np.array(next_states)
    q_values = q_network.predict(states, verbose=0)
    q_next = target_network.predict(next_states, verbose=0)
    for i in range(batch_size):
        if dones[i]:
            q_values[i][actions[i]] = rewards[i]
        else:
            q_values[i][actions[i]] = rewards[i] + gamma * np.max(q_next[i])
    q_network.fit(states, q_values, epochs=1, verbose=0)

# ============================================================
# Main Training Loop
#
# Experimental Modification:
# The baseline implementation stores the environment reward
# directly in experience replay.
#
# This version applies reward shaping before storing the
# experience. Additional penalties are added for:
#
#   - Horizontal displacement from the landing center
#   - Excessive vertical descent speed
#   - Unstable orientation angle
#
# The objective is to encourage more controlled and stable
# landing behavior while keeping the DQN architecture,
# hyperparameters, and training process unchanged.
#
# Original:
# memory.append((state, action, reward, next_state, done))
#
# Modified:
# shaped_reward = shape_reward(next_state, reward)
# memory.append((state, action, shaped_reward, next_state, done))
# ============================================================

for batch in range(num_batches):
    total_rewards = []

    for episode in range(episodes_per_batch):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Apply reward shaping
            shaped_reward = shape_reward(next_state, reward)

            # Store shaped reward instead of raw environment reward
            memory.append(
                (state, action, shaped_reward, next_state, done)
            )

            state = next_state

            # Use original environment reward for reporting
            episode_reward += reward

            train_q_network()

        total_rewards.append(episode_reward)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    target_network.set_weights(q_network.get_weights())

    print(
        f"Batch {batch + 1}/{num_batches}, "
        f"Avg Reward: {np.mean(total_rewards):.2f}"
    )

env.close()
