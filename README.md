# Deep Q-Learning for Emergency Room Simulation

## Objective
The goal of this project is to train a reinforcement learning agent to navigate an emergency room environment and reach the medicine cabinet while avoiding obstacles.
## Instructions for Setting up of project
## Prerequisites
Python 3.8+
## Clone the repository:
git clone https://github.com/Celestin25/DQN.git
## Create a virtual environment:

-python -m venv myenv
-myenv\Scripts\activate

## install the necessary packages:
pip install gym numpy tensorflow 
pip install keras-rl2

## Files
- `emergency_room_env.py`: Custom Gym environment for the emergency room simulation.
- `train_emergency.py`: Script to train the DQN agent.
- `play_emergency.py`: Script to simulate the trained agent.

## How to Run
1. Train the agent:
    ```sh
    python train_emergency.py
    ```
2. Simulate the trained agent:
    ```sh
    python play_emergency.py
    ```

## Video Demonstration
[Link to the 5-minute video demonstration]()

## Results
The trained agent successfully navigates the environment and reaches the medicine cabinet and emergency room while avoiding obstacles.
