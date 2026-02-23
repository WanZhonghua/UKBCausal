import json
import numpy as np
import pandas as pd
from pycirclize import Circos
import matplotlib.cm as cm
import matplotlib.colors as mcolors


npy_path = "/home/wzh/UKB-SJTU/Code/exp2026-01-28T13:42:55.917494/predG.npy"
idx_json_path = "/home/wzh/UKB-SJTU/JSON/all_nodes_index.json"
abbr_json_path = "/home/wzh/UKB-SJTU/JSON/all_nodes_abbrev.json"

with open(idx_json_path, "r", encoding="utf-8") as f:
    name2idx = json.load(f)  

with open(abbr_json_path, "r", encoding="utf-8") as f:
    raw_abbr = json.load(f)  

if any(k in name2idx for k in raw_abbr.keys()):
    fullname2abbr = raw_abbr
else:
    fullname2abbr = {v: k for k, v in raw_abbr.items()}

idx2name = {int(v): k for k, v in name2idx.items()}
n_nodes = len(idx2name)
labels_full = [idx2name[i] for i in range(n_nodes)]
labels = [fullname2abbr.get(x, x) for x in labels_full]


label2idx = {}
for i in range(n_nodes):
    lab = labels[i]
    label2idx[lab] = i


mat = np.load(npy_path)
np.fill_diagonal(mat, 0)
mat = np.nan_to_num(mat, nan=0.0)
mat = np.clip(mat, 0, None)

mat[mat <= 0.2] = 0

df = pd.DataFrame(mat, index=labels, columns=labels)


row_sum = df.sum(axis=1)
col_sum = df.sum(axis=0)
keep = (row_sum > 0) | (col_sum > 0)
df = df.loc[keep, keep]


cmap_blue = cm.get_cmap("Blues")
cmap_green = cm.get_cmap("Greens")
cmap_red = cm.get_cmap("Reds")

T_MIN, T_MAX = 0.25, 0.60

sector_color = {}
for lab in df.index:
    idx = label2idx[lab]
    if 0 <= idx <= 26:  # brain
        t = T_MIN + (T_MAX - T_MIN) * (idx / 26 if 26 else 0)
        sector_color[lab] = mcolors.to_hex(cmap_blue(t))
    elif 27 <= idx <= 31:  # organ
        t = T_MIN + (T_MAX - T_MIN) * ((idx - 27) / 4 if 4 else 0)
        sector_color[lab] = mcolors.to_hex(cmap_green(t))
    else:  # disease 32-42
        t = T_MIN + (T_MAX - T_MIN) * ((idx - 32) / 10 if 10 else 0)
        sector_color[lab] = mcolors.to_hex(cmap_red(t))


def link_kws_handler(from_label: str, to_label: str):
    v = float(df.loc[from_label, to_label])
    if v > 0.5:
        return dict(alpha=0.75, zorder=2)
    else:
        return dict(alpha=0.25, zorder=1)

circos = Circos.chord_diagram(
    df,
    space=1.5,
    r_lim=(92, 100),
    cmap=sector_color,                
    label_kws=dict(r=104, size=8),    
    link_kws=dict(
        direction=1,                   
        ec="none",                    
        lw=0
    ),
    link_kws_handler=link_kws_handler,
)

fig = circos.plotfig(figsize=(8, 8))
fig.savefig("predG_chord_official_arrow_pruned.png", dpi=300, bbox_inches="tight")
print("Saved: predG_chord_official_arrow_pruned.png")
