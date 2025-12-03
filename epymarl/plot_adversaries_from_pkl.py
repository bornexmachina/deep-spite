import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

maddpg = pd.read_pickle('maddpg.pkl')
maddpg = dict(sorted(maddpg.items()))

stage1 = pd.read_pickle('spite_stage1.pkl')
stage1 = dict(sorted(stage1.items()))

stage2 = pd.read_pickle('spite_stage2.pkl')
stage2 = dict(sorted(stage2.items()))

stage3 = pd.read_pickle('spite_stage3.pkl')
stage3 = dict(sorted(stage3.items()))

maddpgkeys = list(maddpg.keys())
_mm = [m.split('_MAC_')[0] for m in maddpgkeys]
_gm = [m.split('_MAC_')[1].split('_')[0] for m in maddpgkeys]

stage1keys = list(stage1.keys())
_ms1 = [m.split('_MAC_')[0] for m in stage1keys]
_gs1 = [m.split('_MAC_')[1].split('_')[0] for m in stage1keys]

stage2keys = list(stage2.keys())
_ms2 = [m.split('_MAC_')[0] for m in stage2keys]
_gs2 = [m.split('_MAC_')[1].split('_')[0] for m in stage2keys]

stage3keys = list(stage3.keys())
_ms3 = [m.split('_MAC_')[0] for m in stage3keys]
_gs3 = [m.split('_MAC_')[1].split('_')[0] for m in stage3keys]

MACS = list(set.intersection(*map(set,[_mm, _ms1, _ms2, _ms3])))
GAMES = list(set.intersection(*map(set,[_gm, _gs1, _gs2, _gs3])))

INDICATORS = ['test_return_mean']

def get_raw_dict_from_mac(data, macname):
    output = {k:v for k, v in data.items() if k.startswith(macname)}
    return output

def get_raw_data_from_game(data, gamename):
    output = [ v for k, v in data.items() if k.split('_MAC_')[1].startswith(gamename)]
    return output

def extract_from_seeds(data, key):
    tmp = {}
    for i in range(len(data)):
        if key in data[i].keys():
            tmp[f'run_{i}'] = data[i][key]        
    output = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in tmp.items() ]))
    output['x'] = list(range(output.shape[0]))
    output['x'] = output['x'] * 500
    return output

def get_key_data(data, key):
    output = extract_from_seeds(data, key)
    return output

def convert_to_len(data):
    for c in data.columns[:-1]:
        try:
            data[c] = data[c].apply(lambda x: len(x))
        except:
            data[c] = 1.
    return data

def sumup_grieve(data):
    for c in data.columns[:-1]:
        try:
            data[c] = data[c].apply(lambda x: np.mean(x['values']))
        except:
            data[c] = 1.
    return data

def get_plot_data(data):
    output = pd.melt(data, id_vars=['x'])
    return output

def get_plot_data_last_50(data):
    data = data.tail(50)
    output = pd.melt(data, id_vars=['x'])
    return output

def plot_data(data, color='grey', label='data'):
    sns.set(rc={'figure.figsize':(15, 5), 'axes.facecolor':'white', 'figure.facecolor':'white'});
    pl = sns.lineplot(data=data, x="x", y="value", color = color, label = label);
    
def overview_plot(maddpg, stage1, stage2, stage3, indicator, gamename):
    plot_data(get_plot_data(get_key_data(get_raw_data_from_game(maddpg, gamename), indicator)), color='black', label='maddpg');
    plot_data(get_plot_data(get_key_data(get_raw_data_from_game(stage1, gamename), indicator)), color='orange', label='stage1');
    plot_data(get_plot_data(get_key_data(get_raw_data_from_game(stage2, gamename), indicator)), color='purple', label='stage2');
    plot_data(get_plot_data(get_key_data(get_raw_data_from_game(stage3, gamename), indicator)), color='red', label='stage3');
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05));
    plt.xlabel('timesteps');
    plt.ylabel(f'{indicator}');
    plt.title(f'{gamename} - {indicator}');
    plt.savefig(f'{gamename}_{indicator}.png', dpi=200);
    plt.show();
    plt.clf();
    
def overview_plot_last_50(maddpg, stage1, stage2, stage3, indicator, gamename):
    plot_data(get_plot_data_last_50(get_key_data(get_raw_data_from_game(maddpg, gamename), indicator)), color='black', label='maddpg');
    plot_data(get_plot_data_last_50(get_key_data(get_raw_data_from_game(stage1, gamename), indicator)), color='orange', label='stage1');
    plot_data(get_plot_data_last_50(get_key_data(get_raw_data_from_game(stage2, gamename), indicator)), color='purple', label='stage2');
    plot_data(get_plot_data_last_50(get_key_data(get_raw_data_from_game(stage3, gamename), indicator)), color='red', label='stage3');
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05));
    plt.xlabel('timesteps');
    plt.ylabel(f'{indicator}');
    plt.title(f'{gamename} - {indicator}');
    plt.savefig(f'{gamename}_{indicator}_last_50.png', dpi=200);
    plt.show();
    plt.clf();
    
def plot_griefers(stage1, stage2, stage3, indicator, gamename):
    plot_data(get_plot_data(convert_to_len(get_key_data(get_raw_data_from_game(stage1, gamename), indicator))), color='orange', label='stage1')
    plot_data(get_plot_data(convert_to_len(get_key_data(get_raw_data_from_game(stage2, gamename), indicator))), color='purple', label='stage2')
    plot_data(get_plot_data(convert_to_len(get_key_data(get_raw_data_from_game(stage3, gamename), indicator))), color='red', label='stage3')
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05));
    plt.xlabel('timesteps');
    plt.ylabel(f'{indicator}');
    plt.title(f'{gamename} - {indicator}');
    plt.savefig(f'{gamename}_{indicator}_griefers.png', dpi=200);
    plt.show();
    plt.clf();
    
def plot_grieve_factor(stage1, stage2, stage3, indicator, gamename):
    plot_data(get_plot_data(sumup_grieve(get_key_data(get_raw_data_from_game(stage1, gamename), indicator))), color='orange', label='stage1')
    plot_data(get_plot_data(sumup_grieve(get_key_data(get_raw_data_from_game(stage2, gamename), indicator))), color='purple', label='stage2')
    plot_data(get_plot_data(sumup_grieve(get_key_data(get_raw_data_from_game(stage3, gamename), indicator))), color='red', label='stage3')
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05));
    plt.xlabel('timesteps');
    plt.ylabel(f'{indicator}');
    plt.title(f'{gamename} - {indicator}');
    plt.savefig(f'{gamename}_{indicator}_grieve_factor.png', dpi=200);
    plt.show();
    plt.clf();

maddpg_avg = defaultdict(dict)
stage1_avg = defaultdict(dict)
stage2_avg = defaultdict(dict)
stage3_avg = defaultdict(dict)
for M in MACS:
    for G in GAMES:
        for I in INDICATORS:
            maddpg_avg[M][G] = get_plot_data(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(maddpg, M), G), I))['value'].mean()
            stage1_avg[M][G] = get_plot_data(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage1, M), G), I))['value'].mean()
            stage2_avg[M][G] = get_plot_data(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage2, M), G), I))['value'].mean()
            stage3_avg[M][G] = get_plot_data(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage3, M), G), I))['value'].mean()
        
MADDPG = pd.DataFrame(maddpg_avg)
MADDPG['learner'] = 'non-spite'
STG1 = pd.DataFrame(stage1_avg)
STG1['learner'] = 'spite_stage1'
STG2 = pd.DataFrame(stage2_avg)
STG2['learner'] = 'spite_stage2'
STG3 = pd.DataFrame(stage3_avg)
STG3['learner'] = 'spite_stage3'

overview = pd.concat([MADDPG, STG1, STG2, STG3])

def barplot_games(data, gamename):
    t = data.loc[gamename]#.plot(kind='bar')
    t = pd.melt(t, id_vars=['learner'])
    t.loc[t.variable == 'timed_attack_maddpg_mac', 'variable'] = 'timed_attack'
    t.loc[t.variable == 'kl_maddpg_mac', 'variable'] = 'kl_div_attack'
    t.loc[t.variable == 'random_attack_maddpg_mac', 'variable'] = 'random_attack'
    t.loc[t.variable == 'maddpg_mac', 'variable'] = 'no_attack'


    colors = ["black", "orange", "purple", "red"]
    # Set your custom color palette
    sns.set(rc={'figure.figsize':(15, 5), 'axes.facecolor':'white', 'figure.facecolor':'white'});
    sns.set_palette(sns.color_palette(colors))
    sns.barplot(data=t, x="variable", y="value", hue="learner")
    plt.legend(loc='right', ncol=1, bbox_to_anchor=(1.12, 0.5));
    plt.xlabel(gamename);
    plt.ylabel(f'expected return');
    plt.savefig(f'{gamename}_attacks.png', dpi=200);
    plt.show();
    plt.clf();

for G in GAMES:
    barplot_games(overview, G)