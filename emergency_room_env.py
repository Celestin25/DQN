import gym
from gym import spaces
import numpy as np
from gym.envs.registration import register
from gym.utils import seeding

register(
    id='EmergencyRoom-v0',
    entry_point='emergency_room_env:EmergencyRoomEnv',
)

class EmergencyRoomEnv(gym.Env):
    def __init__(self):
        super(EmergencyRoomEnv, self).__init__()
        self.grid_size = 5  
        self.action_space = spaces.Discrete(5)  
        self.observation_space = spaces.Box(low=0, high=5, shape=(self.grid_size, self.grid_size), dtype=np.float32)

        # Initial positions
        self.agent_pos = [0, 0]  
        self.emergency_room_pos = [4, 4]  

        
        self.num_doctors = 1
        self.num_nurses = 1
        self.num_beds = 4
        self.doctor_pos = [3, 1]
        self.nurse_pos = [2, 4]
        self.bed_positions = [[1, 1], [1, 3], [3, 3], [4, 1]]

        self.max_steps = 100  
        self.steps = 0

    def _random_positions(self, num):
        positions = []
        while len(positions) < num:
            pos = [self.np_random.integers(0, self.grid_size), self.np_random.integers(0, self.grid_size)]
            if pos != self.agent_pos and pos != self.emergency_room_pos and pos not in positions:
                positions.append(pos)
        return positions

    def reset(self):
        self.agent_pos = [0, 0]  
        self.steps = 0
        self.doctor_pos = self._random_positions(self.num_doctors)[0]
        self.nurse_pos = self._random_positions(self.num_nurses)[0]
        self.bed_positions = self._random_positions(self.num_beds)
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        obs[tuple(self.agent_pos)] = 1 
        obs[tuple(self.emergency_room_pos)] = 2  
        for pos in self.bed_positions:
            obs[tuple(pos)] = 3  
        obs[tuple(self.doctor_pos)] = 4  
        obs[tuple(self.nurse_pos)] = 5  
        return obs

    def step(self, action):
        if action == 0:  # Up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # Down
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2:  # Left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3:  # Right
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
       

        self.steps += 1
        done = self.agent_pos == self.emergency_room_pos or self.steps >= self.max_steps

        if self.agent_pos == self.emergency_room_pos:
            reward = 10
        elif self.agent_pos == self.doctor_pos or self.agent_pos == self.nurse_pos:
            reward = -10
            done = True
        elif self.agent_pos in self.bed_positions:
            reward = -5
        else:
            reward = -1

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '.', dtype=str)
        grid[tuple(self.agent_pos)] = '🙂'
        grid[tuple(self.emergency_room_pos)] = '🏥'
        grid[tuple(self.doctor_pos)] = '🩺'
        grid[tuple(self.nurse_pos)] = '🩺'
        for pos in self.bed_positions:
            grid[tuple(pos)] = '🛏'
        for row in grid:
            print(' '.join(row))
        print(f"Steps: {self.steps}")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
