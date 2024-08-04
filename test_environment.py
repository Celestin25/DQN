from gym.envs.registration import register

register(
    id='EmergencyRoom-v0',
    entry_point='emergency_room_env:EmergencyRoomEnv',
)

import gym

def test_environment():
    try:
        env = gym.make('EmergencyRoom-v0')
        print("Environment created successfully.")
        env.reset()
        print("Environment reset successfully.")
    except Exception as e:
        print(f"Error creating environment: {e}")

test_environment()
