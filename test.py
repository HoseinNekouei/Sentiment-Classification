import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.random.normal(0, 1, self.k)  # True action values
        self.q_estimates = np.zeros(self.k)  # Estimated action values
        self.action_counts = np.zeros(self.k)  # Number of times each action is chosen

    def choose_action(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.k)  # Explore
        else:
            return np.argmax(self.q_estimates)  # Exploit

    def take_action(self, action):
        reward = np.random.normal(self.q_true[action], 1)  # Reward with noise
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]
        return reward

# Initialize bandit and parameters
bandit = Bandit()
num_plays = 1000
num_rounds = 2000
epsilon = 0.0

total_rewards = []

for _ in range(num_rounds):
    total_reward = 0
    for _ in range(num_plays):
        action = bandit.choose_action(epsilon)
        reward = bandit.take_action(action)
        total_reward += reward
    total_rewards.append(total_reward)

print("Average total reward after", num_rounds, "rounds of", num_plays, "plays each:", np.mean(total_rewards))

# Calculate the average total reward for each round
avg_rewards = [np.mean(total_rewards[:i+1]) for i in range(num_rounds)]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_rounds + 1), avg_rewards, color='b', label='Average Total Reward')
plt.xlabel('Round')
plt.ylabel('Average Total Reward')
plt.title('Average Total Reward over Rounds with Epsilon-Greedy Strategy')
plt.legend()
plt.grid(True)
plt.show()