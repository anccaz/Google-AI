# Q-learning algorithm example
# agent takes a random action at each step to explore env

# exmple: initialized CartPole-v1 env for gym library
# demonstrates core interaction loop:
# observation -> action -> reward -> new observation

import gymnasium as gym
import random


def pick_sample_action():
    # In CartPole-v1, action 0 is left, 1 is right
    return random.randint(0, 1)


# Create the CartPole environment
env = gym.make("CartPole-v1")

print("Starting an episode with random actions:")

# An episode is a full run in the environment
for episode in range(1):
    done = False
    # Reset the environment and get the initial state
    state, info = env.reset()
    total_reward = 0
    step_count = 0

    while not done:
        # The agent takes a random action
        action = pick_sample_action()

        # env processes the action and returns the new state, reward,
        # and whether the episode is terminated or truncated
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1
        done = terminated or truncated

        # Optional: Render the environment to visualize the agent's actions
        # env.render()
    print(f"Ep {episode + 1} fin in {step_count} stps w/ trwd: {total_reward}")

# Close the environment after use
env.close()
