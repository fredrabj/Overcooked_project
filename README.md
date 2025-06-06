# Overcooked project

This repository implements a multi agent actor-critic algorithm with centralized training and decentralized execution (CTDE) for the Overcooked-aienvironment.

## Project Goal

The goal of the project is to solve the cramped_room layout in Overcooked-ai and acheive a minimum average reward of 50 points per episode. Each episode has a horizon of 400 steps.

## Structure

Overcooked_project/
├── agents/               # Actor and critic model definitions
├── envs/                 # Overcooked-AI wrapper and environment manager
├── runner.py             # Main training loop
├── utils.py              # Helpers for logging, saving models, etc.
├── train.py              # Script to launch training
└── evaluate.py           # Evaluation script

## Results
Average reward per episode in cramped room: 150
Trained episodes: 3500

Training logs for tensorboard available in the "logs/" folder. "avg_total_reward" is the reward per episode

Checkpoints can be found in the "checkpoints/" folder


## Requirements

overcooked-ai (https://github.com/HumanCompatibleAI/overcooked_ai)
Other requirements are stated in requirements.txt

## Installation

Create a virtual environment, install overcooked-ai from source and install other requirements from requirements.txt

## Usage

To train a model from scratch:
"""python your_script.py --mode train"""

To continue training a model from checkpoint
"""python your_script.py --mode train_resume --checkpoint ckpt-2500 --step 2500"""

In order to evaluate a model, run:
"""python your_script.py --mode eval --checkpoint ckpt-3500 --render"""