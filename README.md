# DQN

# Hospital Emergency Response Simulation with Reinforcement Learning

## Overview

This project demonstrates the use of reinforcement learning to optimize hospital emergency response. The custom environment simulates a hospital emergency room scenario where the agent (hospital staff) must attend to patients while avoiding obstacles and managing resources.


## Environment

### Scenario: Hospital Emergency Response

- **Environment**: A 5x5 grid with various elements:
  - **Patients**: Critical patients needing attention.
  - **Equipment**: Blocks paths.
  - **Staff**: Doctors and nurses as moving obstacles.
  - **Supply Room**: Provides essential supplies.

- **Actions**:
  - **Move Up**: Move the agent up.
  - **Move Down**: Move the agent down.
  - **Move Left**: Move the agent left.
  - **Move Right**: Move the agent right.
  - **Rest**: The agent can choose to stay in place for a turn to simulate strategic resting or waiting.

- **Rewards**:
  - **Attend Patient**: +10 points.
  - **Retrieve Supplies**: +5 points.
  - **Avoid Obstacles**: -10 points for collisions.
  - **Complete All Tasks**: +50 points when all patients are attended.

- **Termination Conditions**:
  - **Success**: All patients are attended to.
  - **Failure**: Collides with an obstacle or runs out of time/steps.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- `gym`
- `numpy`
- `keras`
- `keras-rl2`
- `tensorflow`

You can install the required packages using:

```bash
pip install -r requirements.txt
## Custom Environment

The custom environment is implemented in `emergency_room_env.py`.

## Training the Agent

The script `train_emergency.py` trains the reinforcement learning agent. Run the following command to start training:

```bash
python train_emergency.py


