import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DIR = "/home/wzh/UKB-SJTU/PCA_Selected_D/ori"            
SUMMARY_CSV = "/home/wzh/UKB-SJTU/PCA_Selected_D/pca_summary.csv"
OUTDIR = "/home/wzh/UKB-SJTU/PCA_Selected_D/sum"
os.makedirs(OUTDIR, exist_ok=True)


N_SHOW = 6
SEED = 0  
K_MAX = 6 


def parse_curve(s, sep=";"):
    return np.array([float(x) for x in str(s).split(sep) if x != ""], dtype=float)

def compute_curves_from_roi_csv(fp):
    df1 = pd.read_csv(fp)
    X = df1.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())
    if X.shape[1] < 2:
        return None, None
    Xz = StandardScaler().fit_transform(X.values)
    pca = PCA().fit(Xz)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    return cum, evr


df = pd.read_csv(SUMMARY_CSV)

bestk_map = dict(zip(df["file"], df["pa_k_95"]))
has_curves = ("cum_curve" in df.columns) and ("evr_curve" in df.columns)
rng = np.random.default_rng(SEED)
files = df["file"].tolist()
if len(files) <= N_SHOW:
    chosen = files
else:
    chosen = rng.choice(files, size=N_SHOW, replace=False).tolist()
cum_dict = {}
evr_dict = {}

if has_curves:
    sub = df[df["file"].isin(chosen)]
    for _, row in sub.iterrows():
        name = row["file"]
        cum_dict[name] = parse_curve(row["cum_curve"])
        evr_dict[name] = parse_curve(row["evr_curve"])
else:
    all_csv = glob.glob(os.path.join(DIR, "*.csv"))
    mp = {os.path.splitext(os.path.basename(p))[0]: p for p in all_csv}
    for name in chosen:
        cum, evr = compute_curves_from_roi_csv(mp[name])
        cum_dict[name] = cum
        evr_dict[name] = evr


if len(cum_dict) > 0:
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)

    for name, cum in cum_dict.items():
        y = cum[:min(K_MAX, len(cum))]
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, linewidth=1.8, label=name)

        best_k = int(bestk_map.get(name, 1))
        best_k = max(1, min(best_k, len(cum)))  
        if best_k <= K_MAX:  
            ax.plot(best_k, cum[best_k - 1], marker="*", markersize=12,
                    linestyle="None", zorder=10)

    ax.set_xlabel("Number of Components (k)")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"Cumulative EV")
    ax.grid(True)
    ax.set_xlim(1, K_MAX)
    ax.set_xticks(range(1, K_MAX + 1))
    ax.set_ylim(0, 1.02)

    ax.legend(loc="best", fontsize=8, framealpha=0.75,
              borderpad=0.3, labelspacing=0.25, handlelength=1.5)

    out1 = os.path.join(OUTDIR, f"random{len(cum_dict)}_cum_lines_k10.png")
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)

if len(evr_dict) > 0:
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)

    for name, evr in evr_dict.items():
        y = evr[:min(K_MAX, len(evr))]
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, linewidth=1.8, label=name)

        best_k = int(bestk_map.get(name, 1))
        if 1 <= best_k <= min(K_MAX, len(evr)):
            ax.plot(best_k, evr[best_k - 1], marker="*", markersize=12,
                    linestyle="None", zorder=10)

    ax.set_xlabel("Principal Component (k)")
    ax.set_ylabel("Explained Variance Ratio (EVR)")
    ax.set_title(f"Scree / EVR")
    ax.grid(True)
    ax.set_xlim(1, K_MAX)
    ax.set_xticks(range(1, K_MAX + 1))

    ax.legend(loc="best", fontsize=8, framealpha=0.75,
              borderpad=0.3, labelspacing=0.25, handlelength=1.5)

    out2 = os.path.join(OUTDIR, f"random{len(evr_dict)}_scree_lines_k10.png")
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig)

