# k-armed-bandit-comparison-.-Part-2


Non-Stationary 10-Armed Bandit Project
Overview
This project implements a non-stationary 10-armed bandit problem to study the exploration-exploitation trade-off in dynamic environments. The bandit has 10 arms with time-varying reward distributions N(μi,t,1)
, where initial means μi,1∼N(0,1). Three change scenarios are tested: drift (gradual mean changes), mean-reverting (oscillating means), and abrupt (mean permutation at t=501, with/without reset). Four algorithms are compared: Greedy, Epsilon-Greedy, Optimistic Greedy, and Gradient Bandit, evaluated over 1000 simulations with 2000 steps each. Performance metrics include average per-step reward and optimal action percentage, saved as CSVs, visualized in plots, and summarized in console reports.
The project is part of a Reinforcement Learning assignment (Assignment 1, Part 2) and emphasizes algorithm adaptability in non-stationary settings.
Prerequisites
Python: Version 3.8 or higher (tested in a Windows virtual environment).

Dependencies: Install required libraries using:

bash
pip install numpy>=1.21 matplotlib>=3.4 scipy>=1.7 seaborn>=0.11 tqdm>=4.62 pandas>=2.0

Or create a requirements.txt:
plaintext

numpy>=1.21
matplotlib>=3.4
scipy>=1.7
seaborn>=0.11
tqdm>=4.62
pandas>=2.0

Install with:
bash

pip install -r requirements.txt

Installation
Clone or download the project repository to your local machine.

Navigate to the project directory:
bash

cd path/to/non-stationary-bandit

Set up a virtual environment (optional but recommended):
bash

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/macOS

Install dependencies as above.

Usage
Prepare the Code:
Save the provided code as non_stationary_bandit.py (or your preferred name, e.g., Part 2.py) in the project directory.

Run the Code:
bash

python non_stationary_bandit.py

Execution time: ~15–20 minutes (e.g., 5:44 for abrupt reset scenario).

The code runs experiments for three scenarios: drift, mean-reverting, and abrupt (no reset and reset at t=501).

Outputs:
Console: Pilot study results (e.g., ϵ=0.05, α=0.05 for abrupt), final metrics (average reward, optimal action % over last 500 steps), and detailed reports for each scenario.

CSVs: Four files per scenario (drift_no_reset, mean_reverting_no_reset, abrupt_no_reset, abrupt_reset), e.g.:
avg_rewards_abrupt_no_reset.csv

avg_optimal_action_pct_abrupt_reset.csv

final_rewards_drift_no_reset.csv

avg_regrets_mean_reverting_no_reset.csv

Plots: Three plots per scenario (total expected: 12 PNGs), e.g.:
average_reward_abrupt_reset.png (line plot)

reward_vs_optimal_scatter_drift_no_reset.png (scatter plot)

final_reward_histogram_mean_reverting_no_reset.png (histogram)

Note: Only 11 plots may appear if final_reward_histogram_abrupt_reset.png fails to save.

Optional: Fix Hyperparameters:
To use specific values (e.g., ϵ=0.05, α=0.05), modify run_simulations:
python

best_epsilon = 0.05
best_alpha = 0.05

Place before pilot studies to skip tuning.

File Structure

non-stationary-bandit/
├── non_stationary_bandit.py      # Main code (e.g., Part 2.py)
├── requirements.txt              # Dependency list
├── README.md                     # This file
├── avg_rewards_*.csv             # Average rewards per scenario
├── avg_optimal_action_pct_*.csv  # Optimal action percentages
├── final_rewards_*.csv           # Rewards at t=2000
├── avg_regrets_*.csv             # Cumulative regrets
├── average_reward_*.png          # Reward line plots
├── reward_vs_optimal_scatter_*.png # Reward vs. optimal action scatter plots
├── final_reward_histogram_*.png  # Final reward distribution histograms

Expected Results
Based on provided results (executed on June 18, 2025), key metrics for each scenario (last 500 steps, 1000 simulations):
Drift, No Reset (ϵ=0.1, α=0.05)
Average Reward:
Gradient Bandit: 1.7474

Optimistic Greedy: 1.6299

Epsilon-Greedy: 1.6257

Greedy: 1.1982

Optimal Action %:
Gradient Bandit: 71.32%

Epsilon-Greedy: 68.84%

Optimistic Greedy: 58.77%

Greedy: 32.25%

Mean-Reverting, No Reset (ϵ=0.3, α=0.4)
Average Reward:
Epsilon-Greedy: 0.0018

Greedy: 0.0013

Gradient Bandit: 0.0001

Optimistic Greedy: -0.0018

Optimal Action %:
Optimistic Greedy: 10.07%

Greedy: 9.98%

Epsilon-Greedy: 9.95%

Gradient Bandit: 9.95%

Note: Near-zero rewards suggest means converge, making arms indistinguishable.

Abrupt, No Reset (ϵ=0.05, α=0.05)
Average Reward:
Epsilon-Greedy: 1.2855

Gradient Bandit: 0.9468

Optimistic Greedy: 0.8737

Greedy: 0.6533

Optimal Action %:
Epsilon-Greedy: 58.74%

Gradient Bandit: 36.43%

Optimistic Greedy: 26.91%

Greedy: 20.73%

Abrupt, Hard Reset (ϵ=0.05, α=0.05)
Average Reward:
Gradient Bandit: 1.5109

Epsilon-Greedy: 1.4461

Optimistic Greedy: 1.0259

Greedy: 1.0058

Optimal Action %:
Gradient Bandit: 87.33%

Epsilon-Greedy: 81.57%

Optimistic Greedy: 36.30%

Greedy: 34.10%

Insights:
Gradient Bandit excels in drift and abrupt reset scenarios due to softmax exploration.

Epsilon-Greedy performs well in abrupt no-reset due to random exploration.

Mean-reverting results indicate a potential issue with mean convergence.

Reproducibility
The code is reproducible due to:
Fixed Seeds: BASE_SEED = 42 for bandits/agents, PERM_SEED = 100 for abrupt permutations, and per-arm noise seeds (BASE_SEED + i).

Deterministic Algorithms: Consistent updates for action values and preferences.

Verified Results: Provided outputs match code expectations, confirming reproducibility.

To reproduce:
Use the same Python and library versions.

Run non_stationary_bandit.py without modifying seeds.

Verify outputs against provided results (CSVs, plots, console).

Plots
The code generates 12 plots (3 per scenario: drift_no_reset, mean_reverting_no_reset, abrupt_no_reset, abrupt_reset):
average_reward_{scenario}.png (Line Plot):
Shows average reward over 2000 steps, with a vertical line at t=500 for abrupt scenarios.

E.g., Gradient Bandit peaks at ~1.75 (drift), ~1.51 (abrupt reset).

reward_vs_optimal_scatter_{scenario}.png (Scatter Plot):
Plots reward vs. optimal action % at t=1, 500, 1000, 1500, 2000.

E.g., Gradient Bandit clusters at (~1.51, ~87%) for abrupt reset.

final_reward_histogram_{scenario}.png (Histogram):
Shows reward distribution at t=2000.

E.g., Epsilon-Greedy peaks at ~1.29 for abrupt no-reset.

Note: Only 11 plots were reported, likely missing final_reward_histogram_abrupt_reset.png. Check output directory or debug save errors.
Troubleshooting
Missing Plot:
If final_reward_histogram_abrupt_reset.png is absent, add a debug print in plot_results:
python

print(f"Saving final_reward_histogram_{change_type}_{reset_str}.png")

Ensure disk space and matplotlib backend compatibility (e.g., matplotlib.use('TkAgg')).

Mean-Reverting Anomaly:
Near-zero rewards (~0.0018) suggest means converge to ~0. Adjust κ=0.9\kappa = 0.9\kappa = 0.9
 or noise variance in NonStationaryBandit._initialize_means:
python

kappa = 0.9  # or epsilon_t = [self.noise_rngs[i].normal(0, 0.1) for i in range(self.k)]

Library Mismatches:
Use specified library versions to avoid numerical differences.

Reproducibility Issues:
Do not modify BASE_SEED or PERM_SEED. Back up outputs to avoid overwrites.

Errors:
Check console for exceptions (e.g., file permissions, matplotlib errors). Share logs for assistance.

Contributing
To contribute:
Fork the repository.

Create a feature branch (git checkout -b feature/new-feature).

Commit changes (git commit -m "Add new feature").

Push to the branch (git push origin feature/new-feature).

Open a pull request.

