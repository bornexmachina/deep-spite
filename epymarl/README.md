# Framework
We used epymarl framework for our MARL studies (https://github.com/uoe-agents/epymarl)

# Modifications
We enrich the framework with spiteful agents for MADDPG found in `epymarl/src/learners`, adversarial attacks `epymarl/src/controllers`, as well as several severe bug fixes to actually being able to run the code

# Experiments
We conduct a set of experiments in `lbforaging` and `mpe` environments. Our current experiments are covered by `run_grid.sh`. An enthusiastic reader can easily alter, add or replace the games in the given file