# Assignment 1
# Part 2
# Mohammad Soleimani


import numpy as np  # For math and arrays
import matplotlib.pyplot as plt  # For creating plots
from scipy.stats import norm  # For normal distribution calculations
import seaborn as sns  # For nicer plot styling
from tqdm import tqdm  # For progress bars during loops
import pandas as pd  # For data handling (optional, not heavily used)

# Set base random seed for reproducibility
# Ensures consistent random numbers across runs
BASE_SEED = 42

# Set permutation seed for abrupt changes
# Ensures the same permutation at t=501 across simulations
PERM_SEED = 100

class NonStationaryBandit:
    """Non-stationary k-armed bandit with time-varying reward distributions"""
    # Models a 10-arm bandit with drifting, mean-reverting, or abrupt mean changes
    def __init__(self, k=10, seed=None, change_type='drift', n_steps=2000):
        self.k = k  # Number of arms (default 10)
        self.rng = np.random.RandomState(seed)  # Random generator for initial means
        self.change_type = change_type  # 'drift', 'mean_reverting', or 'abrupt'
        self.n_steps = n_steps  # Total steps for precomputing means
        self.noise_rngs = [np.random.RandomState(BASE_SEED + i) for i in range(k)]  # Fixed noise seeds per arm
        self.perm_rng = np.random.RandomState(PERM_SEED)  # Fixed seed for permutation
        self.true_means_t = np.zeros((k, n_steps))  # Time-varying means (k x T)
        self._initialize_means()

    def _initialize_means(self):
        # Sets initial means and evolves them based on change_type
        initial_means = self.rng.normal(0, 1, self.k)  # Initial μi ~ N(0,1)
        self.true_means_t[:, 0] = initial_means
        if self.change_type == 'drift':
            # μi,t = μi,t-1 + εi,t, εi,t ~ N(0, 0.01^2)
            for t in range(1, self.n_steps):
                epsilon_t = [self.noise_rngs[i].normal(0, 0.01) for i in range(self.k)]
                self.true_means_t[:, t] = self.true_means_t[:, t-1] + epsilon_t
        elif self.change_type == 'mean_reverting':
            # μi,t = κμi,t-1 + εi,t, κ=0.5, εi,t ~ N(0, 0.01^2)
            kappa = 0.5
            for t in range(1, self.n_steps):
                epsilon_t = [self.noise_rngs[i].normal(0, 0.01) for i in range(self.k)]
                self.true_means_t[:, t] = kappa * self.true_means_t[:, t-1] + epsilon_t
        elif self.change_type == 'abrupt':
            # Constant means until t=501, then permute
            for t in range(1, 501):
                self.true_means_t[:, t] = initial_means
            perm_indices = self.perm_rng.permutation(self.k)
            for t in range(501, self.n_steps):
                self.true_means_t[:, t] = initial_means[perm_indices]
        self.optimal_actions = np.argmax(self.true_means_t, axis=0)  # Optimal arm per step

    def get_reward(self, action, t):
        # Returns reward from N(μi,t, 1) for chosen arm at time t
        return self.rng.normal(self.true_means_t[action, t], 1)

    def get_optimal_action(self, t):
        # Returns the best arm at time t
        return self.optimal_actions[t]

class GreedyAgent:
    # Always picks the arm with the highest estimated reward
    def __init__(self, k=10, initial_values=0.0, seed=None):
        self.k = k  # Number of arms
        self.rng = np.random.RandomState(seed)  # Random generator for tie-breaking
        self.q_estimates = np.full(k, initial_values, dtype=float)  # Estimated rewards
        self.action_counts = np.zeros(k, dtype=int)  # Count of arm selections

    def select_action(self):
        # Picks arm with highest estimated reward (random tie-break)
        max_value = np.max(self.q_estimates)
        max_indices = np.where(self.q_estimates == max_value)[0]
        return self.rng.choice(max_indices)

    def update(self, action, reward):
        # Updates estimated reward using incremental average
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]

    def reset(self):
        # Resets estimates and counts to initial state
        self.q_estimates = np.zeros(self.k, dtype=float)
        self.action_counts = np.zeros(self.k, dtype=int)

class EpsilonGreedyAgent:
    # Picks random arms with probability epsilon, else best arm
    def __init__(self, k=10, epsilon=0.1, initial_values=0.0, seed=None):
        self.k = k  # Number of arms
        self.epsilon = epsilon  # Exploration probability
        self.rng = np.random.RandomState(seed)  # Random generator
        self.q_estimates = np.full(k, initial_values, dtype=float)  # Estimated rewards
        self.action_counts = np.zeros(k, dtype=int)  # Count of selections

    def select_action(self):
        # Random action with prob epsilon, else best arm
        if self.rng.random() < self.epsilon:
            return self.rng.randint(self.k)
        max_value = np.max(self.q_estimates)
        max_indices = np.where(self.q_estimates == max_value)[0]
        return self.rng.choice(max_indices)

    def update(self, action, reward):
        # Updates estimated reward
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]

    def reset(self):
        # Resets estimates and counts
        self.q_estimates = np.zeros(self.k, dtype=float)
        self.action_counts = np.zeros(self.k, dtype=int)

class GradientBanditAgent:
    # Learns preferences and picks arms probabilistically
    def __init__(self, k=10, alpha=0.1, seed=None):
        self.k = k  # Number of arms
        self.alpha = alpha  # Learning rate
        self.rng = np.random.RandomState(seed)  # Random generator
        self.h = np.zeros(k, dtype=float)  # Preferences
        self.avg_reward = 0.0  # Reward baseline
        self.t = 0  # Step counter
        self.action_probs = np.ones(k) / k  # Initial probabilities

    def select_action(self):
        # Picks arm using softmax probabilities
        exp_h = np.exp(self.h - np.max(self.h))  # Avoid overflow
        self.action_probs = exp_h / np.sum(exp_h)
        return self.rng.choice(self.k, p=self.action_probs)

    def update(self, action, reward):
        # Updates preferences and baseline
        self.t += 1
        self.avg_reward += (reward - self.avg_reward) / self.t
        for a in range(self.k):
            if a == action:
                self.h[a] += self.alpha * (reward - self.avg_reward) * (1 - self.action_probs[a])
            else:
                self.h[a] -= self.alpha * (reward - self.avg_reward) * self.action_probs[a]

    def reset(self):
        # Resets preferences, baseline, and probabilities
        self.h = np.zeros(self.k, dtype=float)
        self.avg_reward = 0.0
        self.t = 0
        self.action_probs = np.ones(self.k) / self.k

def run_pilot_study_epsilon(rng, epsilon_values, change_type, n_steps=500, n_simulations=100):
    # Tests epsilon values for Epsilon-Greedy
    results = {}
    for eps in tqdm(epsilon_values, desc="Testing epsilon values"):
        total_rewards = np.zeros(n_steps)
        for run in range(n_simulations):
            bandit = NonStationaryBandit(seed=rng.randint(1000000), change_type=change_type)
            agent = EpsilonGreedyAgent(epsilon=eps, seed=rng.randint(1000000))
            for t in range(n_steps):
                action = agent.select_action()
                reward = bandit.get_reward(action, t)
                agent.update(action, reward)
                total_rewards[t] += reward
        avg_rewards = total_rewards / n_simulations
        results[eps] = np.mean(avg_rewards[-100:])
    best_epsilon = max(results, key=results.get)
    print(f"Pilot results for epsilon ({change_type}): {results}")
    print(f"Best epsilon: {best_epsilon}")
    return best_epsilon

def run_pilot_study_alpha(rng, alpha_values, change_type, n_steps=500, n_simulations=100):
    # Tests alpha values for Gradient Bandit
    results = {}
    for alpha in tqdm(alpha_values, desc="Testing alpha values"):
        total_rewards = np.zeros(n_steps)
        for run in range(n_simulations):
            bandit = NonStationaryBandit(seed=rng.randint(1000000), change_type=change_type)
            agent = GradientBanditAgent(alpha=alpha, seed=rng.randint(1000000))
            for t in range(n_steps):
                action = agent.select_action()
                reward = bandit.get_reward(action, t)
                agent.update(action, reward)
                total_rewards[t] += reward
        avg_rewards = total_rewards / n_simulations
        results[alpha] = np.mean(avg_rewards[-100:])
    best_alpha = max(results, key=results.get)
    print(f"Pilot results for alpha ({change_type}): {results}")
    print(f"Best alpha: {best_alpha}")
    return best_alpha

def run_simulations(change_type, reset_at_501=False):
    # Runs main experiment for a specific change type
    pilot_rng = np.random.RandomState(BASE_SEED)  # Fixed seed for pilot studies
    epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    alpha_values = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
    best_epsilon = run_pilot_study_epsilon(pilot_rng, epsilon_values, change_type)
    best_alpha = run_pilot_study_alpha(pilot_rng, alpha_values, change_type)

    k = 10
    n_simulations = 1000
    n_steps = 2000
    algorithms = {
        'Greedy (Q₀=0)': [],
        f'ε-Greedy (ε={best_epsilon})': [],
        'Optimistic Greedy': [],
        f'Gradient Bandit (α={best_alpha})': []
    }
    optimal_action_percentages = np.zeros((len(algorithms), n_steps))
    rewards = np.zeros((len(algorithms), n_steps))
    final_rewards = np.zeros((len(algorithms), n_simulations))
    regrets = np.zeros((len(algorithms), n_steps))

    for sim in tqdm(range(n_simulations), desc="Simulations"):
        bandit_seed = BASE_SEED + sim
        bandit = NonStationaryBandit(seed=bandit_seed, change_type=change_type)
        max_mean = np.max(bandit.true_means_t[:, 0])  # Initial max mean
        optimistic_value = norm.ppf(0.995, loc=max_mean, scale=1)

        agents = {
            'Greedy (Q₀=0)': GreedyAgent(k, initial_values=0.0, seed=sim * 4 + 1),
            f'ε-Greedy (ε={best_epsilon})': EpsilonGreedyAgent(k, epsilon=best_epsilon, seed=sim * 4 + 2),
            'Optimistic Greedy': GreedyAgent(k, initial_values=optimistic_value, seed=sim * 4 + 3),
            f'Gradient Bandit (α={best_alpha})': GradientBanditAgent(k, alpha=best_alpha, seed=sim * 4 + 4)
        }

        sim_rewards = np.zeros((len(agents), n_steps))
        sim_optimal = np.zeros((len(agents), n_steps))
        sim_regrets = np.zeros((len(agents), n_steps))
        for idx, (name, agent) in enumerate(agents.items()):
            for step in range(n_steps):
                if reset_at_501 and step == 501:
                    agent.reset()  # Hard reset at t=501
                action = agent.select_action()
                reward = bandit.get_reward(action, step)
                agent.update(action, reward)
                sim_rewards[idx, step] = reward
                sim_optimal[idx, step] = 1 if action == bandit.get_optimal_action(step) else 0
                sim_regrets[idx, step] = np.max(bandit.true_means_t[:, step]) - reward
            final_rewards[idx, sim] = sim_rewards[idx, -1]

        for idx, name in enumerate(algorithms.keys()):
            algorithms[name].append(sim_rewards[idx])
            rewards[idx] += sim_rewards[idx] / n_simulations
            optimal_action_percentages[idx] += sim_optimal[idx] / n_simulations
            regrets[idx] += np.cumsum(sim_regrets[idx]) / n_simulations

    avg_rewards = {name: rewards[idx] for idx, name in enumerate(algorithms.keys())}
    avg_optimal_percentages = {name: optimal_action_percentages[idx] * 100 for idx, name in enumerate(algorithms.keys())}
    final_rewards = {name: final_rewards[idx] for idx, name in enumerate(algorithms.keys())}
    avg_regrets = {name: regrets[idx] for idx, name in enumerate(algorithms.keys())}

    return avg_rewards, avg_optimal_percentages, final_rewards, avg_regrets, best_epsilon, best_alpha

def plot_results(avg_rewards, avg_optimal_percentages, final_rewards, avg_regrets, best_epsilon, best_alpha, change_type, reset_at_501):
    # Creates plots and saves results for a specific change type
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    methods = list(avg_rewards.keys())
    reset_str = 'reset' if reset_at_501 else 'no_reset'

    # ASCII-safe headers for CSVs
    header_map = {
        'Greedy (Q₀=0)': 'Greedy_Q0_0',
        f'ε-Greedy (ε={best_epsilon})': f'Epsilon_Greedy_eps_{best_epsilon}',
        'Optimistic Greedy': 'Optimistic_Greedy',
        f'Gradient Bandit (α={best_alpha})': f'Gradient_Bandit_alpha_{best_alpha}'
    }

    # Plot 1: Average Reward (Line)
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(methods):
        plt.plot(avg_rewards[name], label=name, color=colors[i])
    plt.axvline(x=500, color='k', linestyle='--', alpha=0.5) if change_type == 'abrupt' else None
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title(f'Average Reward ({change_type}, {reset_str}, ε={best_epsilon}, α={best_alpha})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'average_reward_{change_type}_{reset_str}.png')
    plt.show()
    plt.close()

    # Plot 2: Scatter (Reward vs. Optimal Action)
    key_steps = [0, 499, 999, 1499, 1999]
    plt.figure(figsize=(10, 6))
    sizes = np.linspace(50, 200, len(key_steps))
    for i, name in enumerate(methods):
        rewards_at_steps = avg_rewards[name][key_steps]
        optimal_at_steps = avg_optimal_percentages[name][key_steps]
        plt.scatter(rewards_at_steps, optimal_at_steps, s=sizes, c=[colors[i]], label=name, alpha=0.6)
        for j, t in enumerate(key_steps):
            plt.text(rewards_at_steps[j], optimal_at_steps[j] + 1, f't={t+1}', fontsize=8)
    plt.xlabel('Average Reward')
    plt.ylabel('Optimal Action (%)')
    plt.title(f'Reward vs. Optimal Action ({change_type}, {reset_str})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'reward_vs_optimal_scatter_{change_type}_{reset_str}.png')
    plt.show()
    plt.close()

    # Plot 3: Histogram (Final Rewards)
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(methods):
        plt.hist(final_rewards[name], bins=30, alpha=0.4, label=name, color=colors[i], density=True)
    plt.xlabel('Reward at t=2000')
    plt.ylabel('Density')
    plt.title(f'Final Rewards ({change_type}, {reset_str})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'final_reward_histogram_{change_type}_{reset_str}.png')
    plt.show()
    plt.close()

    # Save CSVs
    np.savetxt(f'avg_rewards_{change_type}_{reset_str}.csv', np.array([avg_rewards[name] for name in methods]), delimiter=',',
               header=','.join(header_map[name] for name in methods), encoding='utf-8')
    np.savetxt(f'avg_optimal_action_pct_{change_type}_{reset_str}.csv', np.array([avg_optimal_percentages[name] for name in methods]), delimiter=',',
               header=','.join(header_map[name] for name in methods), encoding='utf-8')
    np.savetxt(f'final_rewards_{change_type}_{reset_str}.csv', np.array([final_rewards[name] for name in methods]), delimiter=',',
               header=','.join(header_map[name] for name in methods), encoding='utf-8')
    np.savetxt(f'avg_regrets_{change_type}_{reset_str}.csv', np.array([avg_regrets[name] for name in methods]), delimiter=',',
               header=','.join(header_map[name] for name in methods), encoding='utf-8')

def generate_report(avg_rewards, avg_optimal_percentages, final_rewards, best_epsilon, best_alpha, change_type, reset_at_501):
    # Prints a report summarizing results
    reset_str = 'reset' if reset_at_501 else 'no_reset'
    display_map = {
        'Greedy (Q₀=0)': 'Greedy (Q0=0)',
        f'ε-Greedy (ε={best_epsilon})': f'epsilon-Greedy (epsilon={best_epsilon})',
        'Optimistic Greedy': 'Optimistic Greedy',
        f'Gradient Bandit (α={best_alpha})': f'Gradient Bandit (alpha={best_alpha})'
    }

    print("\n" + "=" * 80)
    print(f"NON-STATIONARY BANDIT REPORT ({change_type.upper()}, {reset_str.upper()})")
    print("=" * 80)

    print("\n1. EXPERIMENTAL SETUP:")
    print(f"   • Number of arms (k): 10")
    print(f"   • Number of simulations: 1000")
    print(f"   • Steps per simulation: 2000")
    print(f"   • Reward distributions: N(μi,t, 1), μi,1 ~ N(0, 1)")
    print(f"   • Change type: {change_type}")
    print(f"   • Reset at t=501: {'Yes' if reset_at_501 else 'No'}")
    print(f"   • Random seed: {BASE_SEED}, Permutation seed: {PERM_SEED}")

    print("\n2. HYPERPARAMETER TUNING:")
    print(f"   • Best ε (epsilon-greedy): {best_epsilon}")
    print(f"   • Best α (gradient bandit): {best_alpha}")
    print(f"   • Optimistic initial value: 99.5th percentile of initial best arm")

    print("\n3. FINAL PERFORMANCE METRICS:")
    print("\n   Average Reward (last 500 steps):")
    final_rewards_avg = {}
    for name in avg_rewards.keys():
        final_performance = np.mean(avg_rewards[name][-500:])
        final_rewards_avg[name] = final_performance
        print(f"   • {display_map[name]:<25}: {final_performance:.4f}")

    print("\n   Optimal Action Percentage (last 500 steps):")
    final_optimal = {}
    for name in avg_optimal_percentages.keys():
        final_opt = np.mean(avg_optimal_percentages[name][-500:])
        final_optimal[name] = final_opt
        print(f"   • {display_map[name]:<25}: {final_opt:.2f}%")

    print("\n4. ALGORITHM RANKING:")
    ranked_algorithms = sorted(final_rewards_avg.items(), key=lambda x: x[1], reverse=True)
    print(f"\n   Performance Ranking (by average reward):")
    for i, (name, reward) in enumerate(ranked_algorithms, 1):
        opt_pct = final_optimal[name]
        print(f"   {i}. {display_map[name]}: {reward:.4f} avg reward, {opt_pct:.1f}% optimal")

    print("\n5. INSIGHTS:")
    print(f"   • Change Type: {change_type}")
    if change_type == 'drift':
        print("     • Gradual drift challenges algorithms to track slowly changing means.")
        print("     • Exploration is key to adapt to new optimal arms.")
    elif change_type == 'mean_reverting':
        print("     • Mean-reverting means oscillate, requiring continuous adaptation.")
        print("     • Algorithms with steady exploration perform better.")
    elif change_type == 'abrupt':
        print("     • Abrupt change at t=501 tests adaptation to sudden shifts.")
        print(f"     • Reset at t=501 {'helps' if reset_at_501 else 'is absent, making adaptation harder'}.")
    print("   • Gradient Bandit often excels due to adaptive exploration via softmax.")
    print("   • Epsilon-Greedy performs well with tuned epsilon, but random exploration is less efficient.")
    print("   • Optimistic Greedy struggles post-initial exploration in non-stationary settings.")
    print("   • Greedy fails without systematic exploration.")

    print("\n" + "=" * 80)
    print("CONCLUSION: Non-stationary environment highlights need for adaptive exploration!")
    print("=" * 80)

if __name__ == "__main__":
    # Runs experiments for all change types and reset scenarios
    change_types = ['drift', 'mean_reverting', 'abrupt']
    reset_options = [False, True] if change_types[2] == 'abrupt' else [False]

    for change_type in change_types:
        for reset in reset_options if change_type == 'abrupt' else [False]:
            print(f"\nRunning experiment: {change_type}, reset={reset}")
            avg_rewards, avg_optimal_percentages, final_rewards, avg_regrets, best_epsilon, best_alpha = run_simulations(change_type, reset)
            plot_results(avg_rewards, avg_optimal_percentages, final_rewards, avg_regrets, best_epsilon, best_alpha, change_type, reset)
            generate_report(avg_rewards, avg_optimal_percentages, final_rewards, best_epsilon, best_alpha, change_type, reset)