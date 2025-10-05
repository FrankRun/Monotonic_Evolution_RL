# Monotonic Evolution Reinforcement Learning
:smile: This work has been accepted in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).  
This work implements a novel Reinforcement Learning (RL) approach for autonomous driving with monotonic evolution. The algorithm ensures continuous policy improvement with a high confidence guarantee.

## Project Structure

- `main.py`: Main training script for the reinforcement learning algorithm
- `monotonic_evolution_RL.py`: Implementation of the PPO (Proximal Policy Optimization) algorithm
- `normalization.py`: State and reward normalization utilities
- `replaybuffer.py`: Experience replay buffer for storing transitions
- `VissimEnvironment.py`: Interface between VISSIM traffic simulation and the RL algorithm

## Requirements

- Windows operating system (required for VISSIM integration)
- VISSIM traffic simulation software (version 22)
- Python 3.8
- Conda package manager

## Installation

1. Install VISSIM 22 on your Windows system


2. Create and activate the conda environment:
   ```
   conda env create -f environment.yml
   conda activate monotonic_evolution_rl
   ```

3. Verify VISSIM is properly installed and accessible via COM interface

## Usage

Run the main training script:
```
python main.py
```

You can modify hyperparameters using command line arguments, for example:
```
python main.py --max_train_steps 500000 --gamma 0.98
```

## Configuration

The default hyperparameters can be found in `main.py`. You can customize:
- `--max_train_steps`: Maximum number of training steps
- `--gamma`: Discount factor for future rewards
- `--hidden_width`: Width of hidden layers in networks
- And many other PPO-specific parameters


