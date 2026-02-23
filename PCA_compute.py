import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DIR = "/home/wzh/UKB-SJTU/PCA_Selected_D/ori"
OUTDIR = os.path.join("/home/wzh/UKB-SJTU/PCA_Selected_D/pca_reports")
os.makedirs(OUTDIR, exist_ok=True)

K_MARK = 3

PA_NITER = 100
PA_SEED = 0

def k_for(cum, thr):
    return int(np.searchsorted(cum, thr) + 1)

def parallel_analysis_k(Xz, n_iter=100, seed=0):
    rng = np.random.default_rng(seed)
    n, p = Xz.shape
    real_eigs = PCA().fit(Xz).explained_variance_
    rand_eigs = np.zeros((n_iter, p))
    for i in range(n_iter):
        Z = rng.normal(size=(n, p))
        Z = StandardScaler().fit_transform(Z)
        rand_eigs[i, :] = PCA().fit(Z).explained_variance_
    rand95 = np.percentile(rand_eigs, 95, axis=0)
    return int(np.sum(real_eigs > rand95))

rows = []
csv_files = sorted(glob.glob(os.path.join(DIR, "*.csv")))
print("CSV数量:", len(csv_files))

for fp in csv_files:
    base = os.path.splitext(os.path.basename(fp))[0]
    try:
        df = pd.read_csv(fp)

        X = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.mean())


        Xz = StandardScaler().fit_transform(X.values)
        pca = PCA().fit(Xz)
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)

        k80 = k_for(cum, 0.80)
        pa_k = parallel_analysis_k(Xz, n_iter=PA_NITER, seed=PA_SEED)
        plt.figure()
        plt.plot(range(1, len(cum) + 1), cum, marker="o")
        plt.axvline(K_MARK, linestyle="--")
        if K_MARK <= len(cum):
            plt.scatter([K_MARK], [cum[K_MARK - 1]])
            plt.text(K_MARK, cum[K_MARK - 1], f"  k={K_MARK}, cum={cum[K_MARK - 1]:.3f}")
        for t in [0.80, 0.90, 0.95, 0.99]:
            plt.axhline(t, linestyle="--")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title(f"{base} - Cumulative Explained Variance")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"{base}_cum.png"), dpi=200)
        plt.close()

        plt.figure()
        plt.plot(range(1, len(evr) + 1), evr, marker="o")
        plt.axvline(K_MARK, linestyle="--")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title(f"{base} - Scree Plot (EVR)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"{base}_scree.png"), dpi=200)
        plt.close()

        rows.append({
            "file": base,
            "k_80": k80,
            "pa_k_95": pa_k,
        })

        print(f"[OK] {base}: k_80={k80}, PA={pa_k}")

    except Exception as e:
        print(f"[失败] {base}: {e}")

summary = pd.DataFrame(rows)

if len(summary) > 0:
    summary = summary[["file", "k_80", "pa_k_95"]].sort_values(["pa_k_95", "k_80"])

base='/home/wzh/UKB-SJTU/PCA_Selected_D'
out_csv = os.path.join(base, "pca_summary.csv")
summary.to_csv(out_csv, index=False)

