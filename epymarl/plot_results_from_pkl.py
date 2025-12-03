import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_raw_dict_from_mac(data, macname):
    output = {k:v for k, v in data.items() if k.startswith(macname)}
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

def get_raw_data_from_game(data, gamename):
    output = [ v for k, v in data.items() if k.split('_MAC_')[1].startswith(gamename)]
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
    
def overview_plot(maddpg, stage1, stage2, stage3, indicator, gamename, mac):
    plot_data(get_plot_data(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(maddpg, mac), gamename), indicator)), color='black', label='maddpg');
    plot_data(get_plot_data(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage1, mac), gamename), indicator)), color='orange', label='stage1');
    plot_data(get_plot_data(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage2, mac), gamename), indicator)), color='purple', label='stage2');
    plot_data(get_plot_data(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage3, mac), gamename), indicator)), color='red', label='stage3');
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05));
    plt.xlabel('timesteps');
    plt.ylabel(f'{indicator}');
    plt.title(f'{gamename} - {indicator}');
    plt.savefig(f'{gamename}_{indicator}.png', dpi=200);
    plt.show();
    plt.clf();
    
def overview_plot_last_50(maddpg, stage1, stage2, stage3, indicator, gamename, mac):
    plot_data(get_plot_data_last_50(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(maddpg, mac), gamename), indicator)), color='black', label='maddpg');
    plot_data(get_plot_data_last_50(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage1, mac), gamename), indicator)), color='orange', label='stage1');
    plot_data(get_plot_data_last_50(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage2, mac), gamename), indicator)), color='purple', label='stage2');
    plot_data(get_plot_data_last_50(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage3, mac), gamename), indicator)), color='red', label='stage3');
    plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05));
    plt.xlabel('timesteps');
    plt.ylabel(f'{indicator}');
    plt.title(f'{gamename} - {indicator}');
    plt.savefig(f'{gamename}_{indicator}_last_50.png', dpi=200);
    plt.show();
    plt.clf();
    
def plot_griefers(stage1, stage2, stage3, indicator, gamename, mac):
    plot_data(get_plot_data(convert_to_len(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage1, mac), gamename), indicator))), color='orange', label='stage1')
    plot_data(get_plot_data(convert_to_len(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage2, mac), gamename), indicator))), color='purple', label='stage2')
    plot_data(get_plot_data(convert_to_len(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage3, mac), gamename), indicator))), color='red', label='stage3')
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05));
    plt.xlabel('timesteps');
    plt.ylabel(f'{indicator}');
    plt.title(f'{gamename} - {indicator}');
    plt.savefig(f'{gamename}_{indicator}_griefers.png', dpi=200);
    plt.show();
    plt.clf();
    
def plot_grieve_factor(stage1, stage2, stage3, indicator, gamename, mac):
    plot_data(get_plot_data(sumup_grieve(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage1, mac), gamename), indicator))), color='orange', label='stage1')
    plot_data(get_plot_data(sumup_grieve(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage2, mac), gamename), indicator))), color='purple', label='stage2')
    plot_data(get_plot_data(sumup_grieve(get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(stage3, mac), gamename), indicator))), color='red', label='stage3')
    plt.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05));
    plt.xlabel('timesteps');
    plt.ylabel(f'{indicator}');
    plt.title(f'{gamename} - {indicator}');
    plt.savefig(f'{gamename}_{indicator}_grieve_factor.png', dpi=200);
    plt.show();
    plt.clf();
    
def get_boxplot_data(data, gamename):
    df = get_key_data(get_raw_data_from_game(get_raw_dict_from_mac(data, M), gamename), 'agents_q')
    df = df.dropna()
    frame = []
    for c in df.columns[:-1]:
        tmp = pd.DataFrame(df[c].tolist(), index=df.index)
        tmp = pd.DataFrame(tmp['values'].tolist(), index=tmp.index)
        tmp.columns = [f'agent_{i}' for i in range(len(tmp.columns))]
        frame.append(tmp)
    df = pd.concat(frame)
    return df

def prepare_boxplot_data(maddpg, stage1, stage2, stage3):
    dfmd = get_boxplot_data(maddpg, gamename)
    dfmd['learner'] = 'maddpg'

    dfs1 = get_boxplot_data(stage1, gamename)
    dfs1['learner'] = 'stage1'

    dfs2 = get_boxplot_data(stage2, gamename)
    dfs2['learner'] = 'stage2'

    dfs3 = get_boxplot_data(stage3, gamename)
    dfs3['learner'] = 'stage3'

    plotdf = pd.concat([pd.melt(dfmd, id_vars='learner'),
                        pd.melt(dfs1, id_vars='learner'),
                        pd.melt(dfs2, id_vars='learner'),
                        pd.melt(dfs3, id_vars='learner')])
    return plotdf

def boxplot_agents_q(plotdf, gamename):
    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(15, 5))

    # Plot the orbital period with horizontal boxes
    colors = ["black", "orange", "purple", "red"]
    # Set your custom color palette
    sns.set(rc={'figure.figsize':(15, 5), 'axes.facecolor':'white', 'figure.facecolor':'white'});
    sns.set_palette(sns.color_palette(colors))
    sns.boxplot(x="variable", y="value", hue="learner", data=plotdf, width=.6, showfliers = False)

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="Q values of agents")
    ax.set(xlabel=f'{gamename}')
    ax.get_legend().remove()
    sns.despine(trim=True, left=True)
    plt.savefig(f'{gamename}_agents_q.png', dpi=200);
    plt.show();
    plt.clf();
    
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

INDICATORS = ['return_mean']

M = 'maddpg_mac'

for gamename in GAMES:
    for indicator in INDICATORS:
        overview_plot(maddpg, stage1, stage2, stage3, indicator, gamename, M);
        
for gamename in GAMES:
    for indicator in INDICATORS:
        overview_plot_last_50(maddpg, stage1, stage2, stage3, indicator, gamename, M);
        
for gamename in GAMES:
    plot_griefers(stage1, stage2, stage3, 'griefers', gamename, M);
    
for gamename in GAMES:
    plot_grieve_factor(stage1, stage2, stage3, 'grieve_factor', gamename, M)
    
for gamename in GAMES:
    boxplot_agents_q(prepare_boxplot_data(maddpg, stage1, stage2, stage3), gamename);