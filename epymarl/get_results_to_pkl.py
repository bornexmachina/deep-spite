import os
import pickle
import json
import numpy as np
import pandas as pd
from collections import defaultdict

INDICATORS = ['return_mean', 'q_taken_mean', 'return_std', 'target_mean', 'td_error_abs', 'griefers', 'grieve_factor', 'test_return_mean', 'agents_q']
MPATH = '/1/metrics.json'
CPATH = '/1/config.json'

def extract_values(folder, data):
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]

    for s in subfolders:
        path = s
        path = path + '/' + next(os.walk(path))[1][0]
        path = path + '/' + next(os.walk(path))[1][0]

        c = open(path + CPATH)
        config = json.load(c)

        mac = config['mac']

        m = open(path + MPATH)
        metrics = json.load(m)

        gametag = config['env_args']['key'].split(':')[1]
        seed = config['seed']

        key = mac + '_MAC_' + gametag + '_seed_' + str(seed)

        df = {}
        for i in INDICATORS:
            if i in metrics.keys():
                df[i] = metrics[i]['values']

        data[key] = df

        
maddpg = {}
spite_stage1 = {}
spite_stage2 = {}
spite_stage3 = {}

FOLDERS = ['results/maddpg_learner/', 'results/spiteful_maddpg_learner_stage1/', 'results/spiteful_maddpg_learner_stage2/', 'results/spiteful_maddpg_learner_stage3/']
OUTPUTS = [maddpg, spite_stage1, spite_stage2, spite_stage3]

for F, D in zip(FOLDERS, OUTPUTS):
    extract_values(F, D)

with open('maddpg.pkl', 'wb') as f:
    pickle.dump(maddpg, f)
    
with open('spite_stage1.pkl', 'wb') as f:
    pickle.dump(spite_stage1, f)
    
with open('spite_stage2.pkl', 'wb') as f:
    pickle.dump(spite_stage2, f)
    
with open('spite_stage3.pkl', 'wb') as f:
    pickle.dump(spite_stage3, f)