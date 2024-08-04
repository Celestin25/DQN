import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from emergency_room_env import EmergencyRoomEnv


env = gym.make('EmergencyRoom-v0')  


model = Sequential()
model.add(Input(shape=(env.observation_space.shape[0] * env.observation_space.shape[1],)))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))


model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))


model.build((None, env.observation_space.shape[0] * env.observation_space.shape[1]))


memory = SequentialMemory(limit=10000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, policy=policy, memory=memory, nb_actions=env.action_space.n, nb_steps_warmup=10, gamma=0.99, target_model_update=1e-2)
dqn.compile(Adam(learning_rate=0.001), metrics=['mae'])


dqn.load_weights('dqn_emergency_weights.h5f')


dqn.test(env, nb_episodes=5, visualize=True)
