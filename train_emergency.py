
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


tf.compat.v1.disable_eager_execution()


import register_env

print("Creating environment...")
env = gym.make('EmergencyRoom-v0')
print("Environment created successfully.")


nb_actions = env.action_space.n


obs_shape = env.observation_space.shape
print(f"Observation space shape: {obs_shape}")


model = Sequential([
    Flatten(input_shape=(1, obs_shape[0], obs_shape[1])),  
    Dense(24, activation='relu'),
    Dense(24, activation='relu'),
    Dense(nb_actions, activation='linear')
])


dummy_input = np.random.random((1, 1, obs_shape[0], obs_shape[1])).astype(np.float32)
model(dummy_input)


model.summary()


assert model.output_shape == (None, nb_actions), f"Model output shape {model.output_shape} does not match the number of actions {nb_actions}"


memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=nb_actions, nb_steps_warmup=10, gamma=0.99, target_model_update=1e-2)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])


dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)


dqn.save_weights('dqn_emergency_weights.h5f', overwrite=True)
