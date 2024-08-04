import gym
from gym.envs.registration import register

print("Registering environment...")
register(
    id='EmergencyRoom-v0',
    entry_point='emergency_room_env:EmergencyRoomEnv',
)

print("Environment registered successfully.")

print("Available environments:")
for env_id in gym.envs.registry.keys():
    print(env_id)
