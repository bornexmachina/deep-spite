import os

def split(list_a, chunk_size):
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]

MACS = ['timed_attack_maddpg_mac', 'kl_maddpg_mac', 'random_attack_maddpg_mac', 'maddpg_mac']
LEARNERS = ['maddpg_learner', 'spiteful_maddpg_learner_stage1', 'spiteful_maddpg_learner_stage2', 'spiteful_maddpg_learner_stage3']
CMD = []

for L in LEARNERS:
    directories = os.listdir(f'results/models/{L}/')
    for d in directories:
        key = d.split('_')[2].split('_')[0]
        for M in MACS:
            cmd = (f'python3 src/main.py --config=maddpg --env-config=gymma with env_args.time_limit=25 env_args.key="{key}" learner="{L}" checkpoint_path="./results/models/{L}/{d}/" mac="{M}"')
            CMD.append(cmd)

chunk_size = len(CMD) // 20

SCMD = list(split(CMD, chunk_size))

i = 1
for l in SCMD:
    with open(f'bscript{i}.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(l))
    i = i+1
    
print(f'Number of commands is {len(CMD)}')