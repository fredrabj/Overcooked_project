# Overcooked project

This repository implements a multi agent actor-critic algorithm with centralized training and decentralized execution (CTDE) for the Overcooked-aienvironment.

## Project Goal

The goal of the project is to solve the cramped_room layout in Overcooked-ai and acheive a minimum average reward of 50 points per episode. Each episode has a horizon of 400 steps.

## Structure

Overcooked_project/

├── checkpoints/    # Saved checkpoints  
├── logs/   # Saved training logs  
├── overcooked_a2c.py   # Training and evaluation of model  
├── requirements.txt    
└── README.md

## Results

Average reward per episode in cramped room: 150
Trained episodes: 3500

Training logs for tensorboard available in the "logs/" folder. "avg_total_reward" is the reward per episode

Checkpoints can be found in the "checkpoints/" folder


## Requirements
tensorflow==2.19.0
tensorflow_probability==0.25.0
tf_keras==2.19.0
jinja2
pyyaml==6.0.2
typeguard==4.4.3

opencv_python==4.8.1.78
opencv_contrib_python==4.8.1.78
matplotlib==3.5.1

overcooked-ai (https://github.com/HumanCompatibleAI/overcooked_ai)

## Installation

Create a virtual environment, install requirements from requirements.txt and than install overcooked-ai from source

## Usage

To train a model from scratch:  
```python3 overcooked_a2c.py --mode train```

To continue training a model from checkpoint  
```python3 overcooked_a2c.py --mode train_resume --checkpoint ckpt-2500 --step 2500```

In order to evaluate a model, run:  
```python3 overcooked_a2c.py --mode eval --checkpoint ckpt-3500 --render```

## Notes

The code is based on the lectures and lecture-notes on Actor-Critic methods, tensorflow implementation and multi-agent systems
(https://www.mircomusolesi.org/courses/AAS24-25/AAS24-25-main/) as well as ideas from online sources on multi agent reinforcement learning (https://arxiv.org/pdf/2409.03052).
Some implementation details also took inspiration from this blogpost about PPO (https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

Aids like chatGPT and copilot was occationally used for troubleshooting and help during debugging, and the code might show signs of this. However, I have tried my best to make it my own work and stick to my own ideas and implementations. 