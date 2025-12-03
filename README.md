# Deep Spite - Spiteful Agents in MARL
Existing multi-agent reinforcement learning research commonly assumes rationality of the agents. 
While the design of the stochastic games includes cooperative, competitive and mixed settings, policy design does not consider irrational behaviour harming the agent’s own score.
We propose a novel Nash-suboptimal learning policy, allowing the agents to be myopic regarding their objective goals. 
We introduce the element of spite and analyse its propagation through the evolution of the environment.
Motivated by the conceptual similarity of anti-social and adversarial behaviour, we investigate the effect of spiteful training on robustness to adversarial attacks.

## Overview

Multi Agent Reinforcement Learning differentiate between three settings:
- cooperative, where the agents work towards a common goal
- competitive, where the agents compete against each other towards the goal
- mixed-setting

In each setting a Nash optimality for the agents is expected, basically expecting each agent to maximize its own score,
with regards to its own actions as well as actions of other agents. 

However, Nash optimality might be problematic due to several limitations:
- optimality might be achieved in infinite time horizon
- complete rationality of agents is assumed

We hinge on the second point and propose MARL setting, where agents might decide to actually diminish their returns, 
just to hurt the scores of other agents.

## Project setup & reproducibility

### Installation from scratch on a local machine

0. Clone git repo
1. First create a conda environment from `env.yml` using your favourite conda distribution
2. Enter the new conda environment
3. Install requirements: `pip install -r epymarl/requirements.txt`
4. Install `mpe`: `pip install -e mpe/`

### Installation from scratch on Euler

0. Clone git repo
1. Change directory to `/epymarl` and run `set_up_environment_euler.sh`
2. Install `mpe`: `pip install -e mpe/`

### Reproducing experiments

Please note, that running experiments is computationally exhaustive and might take around 20 hours

0. Load necessary modules `module load gcc/8.2.0` and `module load python_gpu/3.9.9`
1. Change directory to `/epymarl` and run `run_grid.sh`. For several runs (e.g. 5 seeds) re-run the command
2. To initialize adversarial attacks first prepare the commands to run on Euler
    1. From `/epymarl` folder execute `python3 create_cmds_for_adversaries.py` which will create 20 .txt files
    2. From `/epymarl` folder run `run_adversaries.sh` which will batch each of the created files
3. To extract data from logs execute from `/epymarl` folder `python3 get_results_to_pkl.py`
4. To plot average rewards for all games execute from `/epymarl` folder `python3 plot_results_from_pkl.py`
5. To plot effects of adversarial attacks execute from `/epymarl` folder `python3 plot_adversaries_from_pkl.py`
