# PPO LunarLander — Reinforcement Learning Agent

An autonomous lunar landing agent trained using Proximal Policy Optimization (PPO) with Gymnasium and Stable-Baselines3.

The agent learns to control a simulated lunar lander through interaction with the environment, using reward-based learning to develop a landing policy.

## Project Overview

The objective was to train an agent capable of landing a spacecraft safely on a designated landing pad.

The project implements an end-to-end reinforcement learning workflow covering environment setup, parallelized training, policy evaluation, model packaging, and publication to Hugging Face Hub.

## Tech Stack

- Python
- Gymnasium
- Stable-Baselines3
- Proximal Policy Optimization (PPO)
- PyTorch
- Hugging Face Hub
- Google Colab

## Training Configuration

| Parameter | Value |
|---|---:|
| Algorithm | PPO |
| Policy | MlpPolicy |
| Environment | LunarLander-v2 |
| Parallel environments | 16 |
| Training timesteps | 1,000,000 |
| n_steps | 1024 |
| Batch size | 64 |
| Epochs | 4 |
| Gamma | 0.999 |
| GAE lambda | 0.98 |
| Entropy coefficient | 0.01 |

## Evaluation Results

The trained policy was evaluated locally over 100 episodes.

| Metric | Result |
|---|---:|
| Mean reward | 243.04 |
| Standard deviation | 46.13 |
| Local evaluation score | 196.92 |
| Official Hugging Face score | **234.98** |
| Required score | 200 |

**Result: Passed Hugging Face Deep Reinforcement Learning Course — Unit 1.**

The official leaderboard score exceeded the required threshold by 34.98 points.

## Trained Model

The trained model, model card, and gameplay replay are available on Hugging Face:

[View model on Hugging Face](https://huggingface.co/JSI123/ppo-LunarLander-v2)

## Project Workflow

1. Set up the Gymnasium LunarLander environment.
2. Configure PPO using Stable-Baselines3.
3. Train the agent using 16 parallel environments.
4. Evaluate the trained policy over 100 episodes.
5. Package the trained model and generate a gameplay replay.
6. Publish the model to Hugging Face Hub.
7. Validate performance using the official leaderboard.

## Running the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python src/train.py
```

Evaluate the model:

```bash
python src/evaluate.py
```

See the training notebook in `notebooks/` for the complete Colab workflow.

## Learning Outcomes

- Understanding the reinforcement learning agent-environment interaction loop.
- Applying PPO to a continuous simulation task with discrete actions.
- Understanding parallelized environment training.
- Evaluating policies using mean reward and standard deviation.
- Packaging and publishing trained models using Hugging Face Hub.


Environment: Gymnasium LunarLander-v2.
