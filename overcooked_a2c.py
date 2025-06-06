import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random
import datetime
import os
import matplotlib.pyplot as plt
import cv2
import argparse

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

# Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# Hyperparamteres
gamma = 0.99    
actor_lr = 2.5e-4
critic_lr = 1e-3
entropy_coef = 0.001
num_episodes = 3500
gae_lambda = 0.95

# Misc. variables
checkpoint_dir = "checkpoints/a2c"
horizon = 400
layout = "cramped_room"


# Define models

class PolicyNetwork(tf.keras.Model):
    def __init__(self, output_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation="relu")
        self.fc2 = tf.keras.layers.Dense(64, activation="relu")
        self.fc3 = tf.keras.layers.Dense(output_dim, activation="softmax")

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class CentralCritic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation="relu")
        self.fc2 = tf.keras.layers.Dense(64, activation="relu")
        self.value = tf.keras.layers.Dense(1, activation=None)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return tf.squeeze(self.value(x), axis=-1)

# Utility functions
def compute_gae(rewards, values, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t+1] if t + 1 < T else 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def update_policy(policy, optimizer, observations, actions, advantages):
    with tf.GradientTape() as tape:
        logits = policy(observations)
        dist = tfp.distributions.Categorical(probs=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        loss = -tf.reduce_mean(log_probs * advantages + entropy_coef * entropy)
    grads = tape.gradient(loss, policy.trainable_variables)
    grads = [tf.clip_by_norm(g, 0.5) for g in grads]
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    return loss

# Optimised actor forward pass
@tf.function
def compute_action_tf(obs, actor):
    obs = tf.expand_dims(obs, axis=0) 
    probs = actor(obs) 
    dist = tfp.distributions.Categorical(probs=probs)
    action = dist.sample()[0]
    return action

def evaluate(checkpoint, episodes=1, render=False):
    base_mdp = OvercookedGridworld.from_layout_name(layout)
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=400)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    
    agent_1_policy = PolicyNetwork(output_dim)
    agent_2_policy = PolicyNetwork(output_dim)
    critic = CentralCritic()

    # Dummy calls
    _ = agent_1_policy(tf.zeros((1, input_dim)))
    _ = agent_2_policy(tf.zeros((1, input_dim)))
    _ = critic(tf.zeros((1, 2 * input_dim)))

    actor_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    actor_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    # Restore from checkpoint
    ckpt = tf.train.Checkpoint(
        actor1=agent_1_policy,
        actor2=agent_2_policy,
        critic=critic,
        opt1=actor_optimizer_1,
        opt2=actor_optimizer_2,
        crit_opt=critic_optimizer
    )

    ckpt.restore(os.path.join(checkpoint_dir, checkpoint)).expect_partial()

    state = env.reset()
    done = False
    rewards = []

    for i in range(episodes):
        state = env.reset()
        total_reward = 0
        while not done:
            obs1, obs2 = state["both_agent_obs"]
            obs1_tensor = tf.convert_to_tensor(obs1, dtype=tf.float32)
            obs2_tensor = tf.convert_to_tensor(obs2, dtype=tf.float32)

            action1_tf = compute_action_tf(obs1_tensor, agent_1_policy)
            action2_tf = compute_action_tf(obs2_tensor, agent_2_policy)

            action1 = int(action1_tf.numpy())
            action2 = int(action2_tf.numpy())


            state, reward, done, _ = env.step((action1, action2))
            total_reward += reward

            if render:
                image = env.render()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, (400,400))
                cv2.imshow("Overcooked", image)
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break

        rewards.append(total_reward)

    print(f"Evaluation episode completed. Total reward: {np.mean(np.array(total_reward))}")
    env.close()


def train(checkpoint=None, checkpoint_step=0):

    # Set up the environment
    base_mdp = OvercookedGridworld.from_layout_name(layout)
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=horizon)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Init models and optimizers
    agent_1_policy = PolicyNetwork(output_dim)
    agent_2_policy = PolicyNetwork(output_dim)
    critic = CentralCritic()

    actor_optimizer_1 = tf.keras.optimizers.Adam(actor_lr)
    actor_optimizer_2 = tf.keras.optimizers.Adam(actor_lr)
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

    # Dummy calls
    dummy_input = tf.zeros((1, input_dim), dtype=tf.float32)
    _ = agent_1_policy(dummy_input)
    _ = agent_2_policy(dummy_input)
    _ = critic(tf.concat([dummy_input, dummy_input], axis=-1))

    
    # Restore from checkpoint
    if not checkpoint is None:
        ckpt = tf.train.Checkpoint(
            actor1=agent_1_policy,
            actor2=agent_2_policy,
            critic=critic,
            opt1=actor_optimizer_1,
            opt2=actor_optimizer_2,
            crit_opt=critic_optimizer
        )

        manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, max_to_keep=5)
        ckpt.restore(os.path.join(checkpoint_dir, checkpoint)).expect_partial()
    
    # Logging setup.
    episode = checkpoint_step
    history = np.zeros((num_episodes, 3))
    log_dir = "logs/centre_pots/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    agent_1_writer = tf.summary.create_file_writer(log_dir + "/agent_1")
    agent_2_writer = tf.summary.create_file_writer(log_dir + "/agent_2")
    os.makedirs(checkpoint_dir, exist_ok=True)
    ep_reward_tot = 0

    while episode < num_episodes:
        state = env.reset()
        done = False

        obs1_list, act1_list, rew1_list = [], [], []
        obs2_list, act2_list, rew2_list = [], [], []

        # Run the episode
        while not done:
            obs1, obs2 = state["both_agent_obs"]
            obs1_tensor = tf.convert_to_tensor(obs1, dtype=tf.float32)
            obs2_tensor = tf.convert_to_tensor(obs2, dtype=tf.float32)

            action1_tf = compute_action_tf(obs1_tensor, agent_1_policy)
            action2_tf = compute_action_tf(obs2_tensor, agent_2_policy)

            action1 = int(action1_tf.numpy())
            action2 = int(action2_tf.numpy())

            state, _, done, info = env.step((action1, action2))

            sparse_r1, sparse_r2 = info["sparse_r_by_agent"]
            shaped_r1, shaped_r2 = info["shaped_r_by_agent"]
            
            rew1 = sparse_r1 + shaped_r1
            rew2 = sparse_r2 + shaped_r2

            ep_reward_tot += sparse_r1 + sparse_r2

            obs1_list.append(obs1)
            act1_list.append(action1)
            rew1_list.append(rew1)

            obs2_list.append(obs2)
            act2_list.append(action2)
            rew2_list.append(rew2)

        # Processing after episode end

        obs1_tensor = tf.convert_to_tensor(obs1_list, dtype=tf.float32)
        obs2_tensor = tf.convert_to_tensor(obs2_list, dtype=tf.float32)
        joint_obs = tf.concat([obs1_tensor, obs2_tensor], axis=-1)
        rewards = np.array(rew1_list) + np.array(rew2_list)

        # Pass the joint observation through the critic and calulate advantage
        values = critic(joint_obs)
        advantages, returns = compute_gae(rewards, values, gamma, gae_lambda)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

        # Update the critic.
        with tf.GradientTape() as tape:
            value_estimates = critic(joint_obs)
            critic_loss = tf.reduce_mean(tf.square(returns - value_estimates))
        critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables))

        # Convert lists of actions to tensors.
        act1_tensor = tf.convert_to_tensor(act1_list, dtype=tf.int32)
        act2_tensor = tf.convert_to_tensor(act2_list, dtype=tf.int32)
        
        # Update agent policies
        loss1 = update_policy(agent_1_policy, actor_optimizer_1, obs1_tensor, act1_tensor, advantages)
        loss2 = update_policy(agent_2_policy, actor_optimizer_2, obs2_tensor, act2_tensor, advantages)

        

        # Logging
        ep_reward_1 = sum(rew1_list)
        ep_reward_2 = sum(rew2_list)
        history[episode] = np.array([ep_reward_1, ep_reward_2, ep_reward_tot])
        window = min(30, episode + 1)
        avg_reward_1 = np.mean(history[episode - window + 1:episode + 1, 0])
        avg_reward_2 = np.mean(history[episode - window + 1:episode + 1, 1])
        avg_reward_tot = np.mean(history[episode - window + 1: episode + 1, 2])
        print(f"Episode {episode+1}: Reward Agent 1 = {ep_reward_1:.2f} (Avg: {avg_reward_1:.2f}), "
              f"Agent 2 = {ep_reward_2:.2f} (Avg: {avg_reward_2:.2f}) | Total: {ep_reward_tot:.2f} (Avg: {avg_reward_tot:.2f})")
        ep_reward_tot = 0

        # Tensorboard logging
        with agent_1_writer.as_default():
            tf.summary.scalar("policy_loss", loss1, step=episode)
            tf.summary.scalar("episode_return", sum(rew1_list), step=episode)
            tf.summary.scalar("avg_reward", avg_reward_1, step=episode)
        with agent_2_writer.as_default():
            tf.summary.scalar("policy_loss", loss2, step=episode)
            tf.summary.scalar("episode_return", sum(rew2_list), step=episode)
            tf.summary.scalar("avg_reward", avg_reward_2, step=episode)
            tf.summary.scalar("avg_total_reward", avg_reward_tot, step=episode)
        episode += 1

        # Save chackpoints
        if episode % 500 == 0:
            ckpt = tf.train.Checkpoint(actor1=agent_1_policy, actor2=agent_2_policy,
                           critic=critic, opt1=actor_optimizer_1,
                           opt2=actor_optimizer_2, crit_opt=critic_optimizer)
        
            manager = tf.train.CheckpointManager(ckpt, directory=checkpoint_dir, max_to_keep=5)
            manager.save(checkpoint_number=episode)

    env.close()

    # Plot results
    window_size = 30
    moving_avg_0 = np.convolve(history[:,0], np.ones(window_size)/window_size, mode='valid')
    moving_avg_1 = np.convolve(history[:,1], np.ones(window_size)/window_size, mode='valid')
    moving_avg_2 = np.convolve(history[:,2], np.ones(window_size)/window_size, mode='valid')

    plt.plot(np.arange(window_size-1, len(history[:,0])), moving_avg_0, label='Agent 1 (Avg Last 30)')
    plt.plot(np.arange(window_size-1, len(history[:,1])), moving_avg_1, label='Agent 2 (Avg Last 30)')
    plt.plot(np.arange(window_size-1, len(history[:,2])), moving_avg_2, label='Total reward (Avg Last 30)')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MAPPO training or evaluation.")
    parser.add_argument(
        "--mode", choices=["train", "train_resume", "eval"], required=True,
        help="Mode to run: 'train' to start from scratch, 'train_resume' to resume from a checkpoint, 'eval' to evaluate a checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint name (e.g., 'ckpt-2500'). Required for 'train_resume' and 'eval'."
    )
    parser.add_argument(
        "--step", type=int, default=0,
        help="Step number associated with the checkpoint. Required for 'train_resume'."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render environment during evaluation."
    )

    args = parser.parse_args()


    # Select the preferred mode
    if args.mode == "train":
        train()

    elif args.mode == "train_resume":
        if not args.checkpoint or args.step <= 0:
            raise ValueError("You must provide a valid --checkpoint and --step for resuming training.")
        train(args.checkpoint, args.step)

    elif args.mode == "eval":
        if not args.checkpoint:
            raise ValueError("You must provide a valid --checkpoint for evaluation.")
        evaluate(args.checkpoint, render=args.render)