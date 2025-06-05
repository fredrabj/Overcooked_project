import tensorflow as tf
import numpy as np
import random
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import datetime
import time

# Seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Hyperparameters
gamma = 0.99
actor_lr = 2.5e-4
critic_lr = 1e-3
entropy_coef = 0.01
num_episodes = 3000

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
        return tf.squeeze(self.value(x), axis=-1)  # shape: (batch,)

    
def compute_returns(rewards, gamma):
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns)
    return (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

def update_policy(policy, optimizer, observations, actions, returns):
    with tf.GradientTape() as tape:
        logits = policy(observations)
        action_probs = tf.gather(logits, actions, axis=1, batch_dims=1)
        log_probs = tf.math.log(action_probs + 1e-8)
        entropy = -tf.reduce_sum(logits * tf.math.log(logits + 1e-8), axis=1)
        loss = -tf.reduce_mean(log_probs * returns + entropy_coef * entropy)
    grads = tape.gradient(loss, policy.trainable_variables)
    grads = [tf.clip_by_norm(g, 0.5) for g in grads]
    optimizer.apply_gradients(zip(grads, policy.trainable_variables))
    return loss

# Optimized forward-pass
@tf.function
def compute_action_tf(obs, actor):
    obs = tf.expand_dims(obs, axis=0)  # shape (1, obs_dim)
    probs = actor(obs)  # shape (1, num_actions), must be a Tensor
    dist = tfp.distributions.Categorical(probs=probs)
    action = dist.sample()[0]  # remove batch dim
    return action  # optionally also return log_prob


def main():
    base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=400)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent_1_policy = PolicyNetwork(output_dim)
    agent_2_policy = PolicyNetwork(output_dim)
    critic = CentralCritic()

    actor_optimizer_1 = tf.keras.optimizers.Adam(actor_lr)
    actor_optimizer_2 = tf.keras.optimizers.Adam(actor_lr)
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

    # Initialize trained variables
    dummy_input = tf.random.uniform((1, input_dim), dtype=tf.float32)
    _ = agent_1_policy(dummy_input)
    _ = agent_2_policy(dummy_input)

    history = np.zeros((num_episodes, 2))
    train_step = 0


    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    agent_1_writer = tf.summary.create_file_writer(log_dir + "/agent_1")
    agent_2_writer = tf.summary.create_file_writer(log_dir + "/agent_2")


    for episode in range(num_episodes):
        state = env.reset()
        done = False

        obs1_list, act1_list, rew1_list = [], [], []
        obs2_list, act2_list, rew2_list = [], [], []

        while not done:
            obs1, obs2 = state["both_agent_obs"]
            obs1_tensor = tf.convert_to_tensor(obs1, dtype=tf.float32)
            obs2_tensor = tf.convert_to_tensor(obs2, dtype=tf.float32)

            action1_tf = compute_action_tf(obs1_tensor, agent_1_policy)
            action2_tf = compute_action_tf(obs2_tensor, agent_2_policy)

            action1 = int(action1_tf.numpy())
            action2 = int(action2_tf.numpy())

            state, reward, done, info = env.step((action1, action2))
            if reward != 0:
                print("== SOup delivered! ==")
            rew1, rew2 = info["shaped_r_by_agent"]
            rew1 += reward
            rew2 += reward

            obs1_list.append(obs1)
            act1_list.append(action1)
            rew1_list.append(rew1)

            obs2_list.append(obs2)
            act2_list.append(action2)
            rew2_list.append(rew2)

        obs1_tensor = tf.convert_to_tensor(obs1_list, dtype=tf.float32)
        obs2_tensor = tf.convert_to_tensor(obs2_list, dtype=tf.float32)
        joint_obs = tf.concat([obs1_tensor, obs2_tensor], axis=-1)  # shape (T, 2 * obs_dim)

        rewards = np.array(rew1_list) + np.array(rew2_list)  # shared reward
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)

        last_value = tf.squeeze(critic(joint_obs[-1:]))
        values = critic(joint_obs)
        next_values = tf.concat([values[1:], [last_value]], axis=0)

        td_target = rewards_tensor + gamma * next_values
        advantages = td_target - values

        with tf.GradientTape() as tape:
            value_estimates = critic(joint_obs)
            critic_loss = tf.reduce_mean(tf.square(td_target - value_estimates))
        grads = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

        # Policy updates using shared advantages
        act1_tensor = tf.convert_to_tensor(act1_list, dtype=tf.int32)
        act2_tensor = tf.convert_to_tensor(act2_list, dtype=tf.int32)

        loss1 = update_policy(agent_1_policy, actor_optimizer_1, obs1_tensor, act1_tensor, advantages)
        loss2 = update_policy(agent_2_policy, actor_optimizer_2, obs2_tensor, act2_tensor, advantages)

        # # Log to TensorBoard
        with agent_1_writer.as_default():
            tf.summary.scalar("policy_loss", loss1, step=train_step)

        with agent_2_writer.as_default():
            tf.summary.scalar("policy_loss", loss2, step=train_step)

        train_step += 1

        # Logging
        ep_reward_1 = sum(rew1_list)
        ep_reward_2 = sum(rew2_list)

        with agent_1_writer.as_default():
            tf.summary.scalar("episode_return", ep_reward_1, step=train_step)

        with agent_2_writer.as_default():
            tf.summary.scalar("episode_return", ep_reward_2, step=train_step)

        history[episode] = np.array([ep_reward_1, ep_reward_2])

        # Compute moving average of last 10 episodes
        window = min(30, episode + 1)
        avg_reward_1 = np.mean(history[episode - window + 1:episode + 1, 0])
        avg_reward_2 = np.mean(history[episode - window + 1:episode + 1, 1])

        print(f"Episode {episode+1}: "
            f"Reward Agent 1 = {ep_reward_1:.2f} (Avg: {avg_reward_1:.2f}), "
            f"Agent 2 = {ep_reward_2:.2f} (Avg: {avg_reward_2:.2f})")

    env.close()
    # Compute moving averages for plotting
    def moving_average(data, window=30):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.plot(moving_average(history[:, 0]), label='Agent 1 (Avg Last 30)')
    plt.plot(moving_average(history[:, 1]), label='Agent 2 (Avg Last 30)')
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Average Rewards Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()