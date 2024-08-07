import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from emergency_room_env import EmergencyRoomEnv
from keras.layers import Dense, Input, Flatten

# Create the environment
env = gym.make('EmergencyRoom-v0')

# Build the model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))  # Flatten the input
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Build the model with the correct input shape
model.build((None, 1) + env.observation_space.shape)

# Set up memory and policy
memory = SequentialMemory(limit=10000, window_length=1)
policy = EpsGreedyQPolicy()

# Set up the DQN agent
dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=env.action_space.n, nb_steps_warmup=10, gamma=0.99, target_model_update=1e-2)
dqn.compile(Adam(learning_rate=0.001), metrics=['mae'])

# Load weights
dqn.load_weights('dqn_emergency_weights.h5f')

# Test the agent
dqn.test(env, nb_episodes=5, visualize=True)
