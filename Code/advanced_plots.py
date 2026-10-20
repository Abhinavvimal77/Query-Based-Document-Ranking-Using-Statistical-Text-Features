"""
Advanced & 3D visualisations for Project 23 (Query-Based Document Ranking).
Put this file in the SAME folder as doc_ranking.py and run:
    python advanced_plots.py          (real Cranfield data)
    python advanced_plots.py --toy    (test only)
Figures are saved in results/adv_*.png
"""
import sys, traceback
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from mpl_toolkits.mplot3d import Axes3D  # noqa
import matplotlib.pyplot as plt
from doc_ranking import *   # loaders, Index, scoring functions, metrics, OUT

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold"})
toy = "--toy" in sys.argv
docs, queries, qrels = load_toy() if toy else load_cranfield()
ix = Index(docs, queries)
DIM = min(200, ix.vocab_size - 1)

S = {"TF": score_tf(ix), "TF-IDF": score_tfidf(ix), "BM25": score_bm25(ix),
     "QL-Dir": score_qlm(ix), "LSA": score_lsa(ix, dims=DIM), "Hybrid": score_hybrid(ix, dims=DIM)}
names = list(S)
pq = {n: per_query_metrics(S[n], ix, qrels) for n in names}
metrics = ["P@10", "R@10", "MAP", "MRR", "nDCG@10"]
summary = pd.DataFrame({n: pq[n].mean() for n in names}).T[metrics]
colors = plt.cm.tab10(np.arange(len(names)))
save = lambda f: (plt.tight_layout(), plt.savefig(f"{OUT}/{f}", dpi=200, bbox_inches="tight"), plt.close(), print("saved", f))

def safe(fn):
    try: fn()
    except Exception: print("FAILED:", fn.__name__); traceback.print_exc()

# ---- 1. 3D bar chart: models x metrics
def fig1():
    fig = plt.figure(figsize=(11, 7)); ax = fig.add_subplot(111, projection="3d")
    for i, n in enumerate(names):
        ax.bar3d(np.arange(len(metrics)), np.full(len(metrics), i), 0, 0.55, 0.55,
                 summary.loc[n].values, color=colors[i], shade=True, alpha=0.9)
    ax.set_xticks(np.arange(len(metrics)) + 0.27); ax.set_xticklabels(metrics)
    ax.set_yticks(np.arange(len(names)) + 0.27); ax.set_yticklabels(names)
    ax.set_zlabel("Score"); ax.set_title("3D comparison of ranking models across metrics"); ax.view_init(28, -55)
    save("adv_01_3d_bar_models_metrics.png")

# ---- 2. 3D surface: BM25 MAP over k1 x b
def fig2():
    k1s = np.linspace(0.4, 3.6, 10); bs = np.linspace(0, 1, 9)
    Z = np.array([[per_query_metrics(score_bm25(ix, k, b), ix, qrels)["MAP"].mean() for b in bs] for k in k1s])
    B, K1 = np.meshgrid(bs, k1s)
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(B, K1, Z, cmap="viridis", edgecolor="k", linewidth=0.3, alpha=0.95)
    i, j = np.unravel_index(Z.argmax(), Z.shape)
    ax.scatter(bs[j], k1s[i], Z[i, j], color="red", s=80, label=f"best MAP={Z[i,j]:.3f} (k1={k1s[i]:.1f}, b={bs[j]:.2f})")
    ax.set_xlabel("b (length normalisation)"); ax.set_ylabel("k1 (tf saturation)"); ax.set_zlabel("MAP")
    ax.set_title("BM25 parameter surface"); fig.colorbar(surf, shrink=0.6, pad=0.1); ax.legend(loc="upper left"); ax.view_init(30, -125)
    save("adv_02_3d_surface_bm25.png")

# ---- 3. 3D surface: Hybrid MAP over LSA dims x alpha
def fig3():
    dims = sorted({d for d in [25, 50, 100, 150, 200, 300, 400] if d < ix.vocab_size - 1} or
                  {max(2, ix.vocab_size // 4), max(3, ix.vocab_size // 2)})
    alphas = np.round(np.linspace(0, 1, 11), 1)
    bm = minmax(score_bm25(ix, 2.0, 0.75)); Z = np.zeros((len(dims), len(alphas)))
    for i, d in enumerate(dims):
        ls = minmax(score_lsa(ix, dims=d))
        for j, a in enumerate(alphas):
            Z[i, j] = per_query_metrics(a * bm + (1 - a) * ls, ix, qrels)["MAP"].mean()
    A, D = np.meshgrid(alphas, dims)
    fig = plt.figure(figsize=(9, 7)); ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(A, D, Z, cmap="plasma", edgecolor="k", linewidth=0.3)
    i, j = np.unravel_index(Z.argmax(), Z.shape)
    ax.scatter(alphas[j], dims[i], Z[i, j], color="cyan", s=90, edgecolor="k", label=f"best MAP={Z[i,j]:.3f} (dims={dims[i]}, a={alphas[j]})")
    ax.set_xlabel("alpha (weight of BM25)"); ax.set_ylabel("LSA dimensions"); ax.set_zlabel("MAP")
    ax.set_title("Hybrid model: fusion weight vs LSA dimensionality"); fig.colorbar(surf, shrink=0.6, pad=0.1)
    ax.legend(loc="upper left"); ax.view_init(30, -130)
    save("adv_03_3d_surface_hybrid.png")

# ---- 4. 3D document space (LSA, 3 components) for one query
def fig4():
    idf = np.log((ix.N + 1) / (ix.df + 1)) + 1
    tf = ix.X.copy(); tf.data = 1 + np.log(tf.data)
    A = normalize(tf @ sp.diags(idf))
    svd = TruncatedSVD(n_components=3, random_state=42); P = svd.fit_transform(A)
    cand = [q for q in qrels if q in ix.q_ids]
    q = max(cand, key=lambda x: len(qrels[x])) if toy else sorted(cand, key=lambda x: -len(qrels[x]))[len(cand) // 20]
    qi = ix.q_ids.index(q); qv = svd.transform(normalize(ix.Q[qi] @ sp.diags(idf)))
    rel = np.array([d in qrels[q] for d in ix.doc_ids]); top = np.argsort(-S["BM25"][qi])[:10]
    fig = plt.figure(figsize=(10, 7.5)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*P[~rel].T, c="lightgrey", s=8, alpha=0.5, label="other documents")
    ax.scatter(*P[rel].T, c="red", s=70, marker="o", edgecolor="k", label=f"relevant ({rel.sum()})")
    ax.scatter(*P[top].T, c="royalblue", s=60, marker="^", label="BM25 top-10")
    ax.scatter(*qv.T, c="gold", s=300, marker="*", edgecolor="k", label="query")
    ax.set_xlabel("LSA dim 1"); ax.set_ylabel("LSA dim 2"); ax.set_zlabel("LSA dim 3")
    ax.set_title(f"3D latent semantic space - query {q}"); ax.legend(loc="upper left"); ax.view_init(22, 40)
    save("adv_04_3d_lsa_document_space.png")

# ---- 5. Radar chart
def fig5():
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(7.5, 7.5)); ax = fig.add_subplot(111, polar=True)
    for i, n in enumerate(names):
        v = summary.loc[n].tolist() + [summary.loc[n].iloc[0]]
        ax.plot(ang, v, color=colors[i], lw=2, label=n); ax.fill(ang, v, color=colors[i], alpha=0.07)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(metrics); ax.set_title("Radar chart of model performance", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1)); save("adv_05_radar_models.png")

# ---- 6. Per-query AP heatmap + violin
def fig6():
    M = pd.DataFrame({n: pq[n]["MAP"] for n in names}); M = M.loc[M.mean(1).sort_values().index]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.2]})
    im = ax[0].imshow(M.values, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=35); ax[0].set_ylabel("queries (sorted by mean AP)")
    ax[0].set_title("Per-query Average Precision"); plt.colorbar(im, ax=ax[0], fraction=0.05)
    vp = ax[1].violinplot([M[n] for n in names], showmeans=True, showmedians=False)
    for b, c in zip(vp["bodies"], colors): b.set_facecolor(c); b.set_alpha(0.7)
    ax[1].set_xticks(range(1, len(names) + 1)); ax[1].set_xticklabels(names, rotation=35); ax[1].set_ylabel("AP"); ax[1].set_title("AP distribution (mean marked)")
    save("adv_06_heatmap_violin_per_query.png")

# ---- 7. BM25 vs LSA scatter (win/loss)
def fig7():
    a, b = pq["BM25"]["MAP"], pq["LSA"]["MAP"]; win = b > a + 1e-9; lose = a > b + 1e-9
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(a[win], b[win], c="seagreen", alpha=0.7, label=f"LSA better ({win.sum()})")
    plt.scatter(a[lose], b[lose], c="crimson", alpha=0.7, label=f"BM25 better ({lose.sum()})")
    plt.scatter(a[~win & ~lose], b[~win & ~lose], c="grey", alpha=0.5, label=f"tie ({(~win & ~lose).sum()})")
    plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("BM25 AP"); plt.ylabel("LSA AP"); plt.title("Per-query comparison: BM25 vs LSA"); plt.legend()
    save("adv_07_scatter_bm25_vs_lsa.png")

# ---- 8. Recall@k curves
def fig8():
    ks = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 400]
    plt.figure(figsize=(8, 5.5))
    for i, n in enumerate(names):
        rec = []
        for k in ks:
            r = []
            for qi, q in enumerate(ix.q_ids):
                if q not in qrels: continue
                top = set(ix.doc_ids[j] for j in np.argsort(-S[n][qi])[:k]); r.append(len(top & set(qrels[q])) / len(qrels[q]))
            rec.append(np.mean(r))
        plt.plot(ks, rec, marker="o", ms=4, color=colors[i], label=n)
    plt.xscale("log"); plt.xlabel("k (number of retrieved documents)"); plt.ylabel("Mean Recall@k"); plt.grid(alpha=0.3)
    plt.title("Recall@k curves"); plt.legend(); save("adv_08_recall_at_k.png")

# ---- 9. 3D query difficulty scatter
def fig9():
    q = pq["BM25"].index
    ql = np.array([len(preprocess(queries[x])) for x in q]); nr = np.array([len(qrels[x]) for x in q])
    ap = pq["BM25"]["MAP"].values; gain = pq["LSA"]["MAP"].values - ap
    fig = plt.figure(figsize=(10, 7.5)); ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(ql, nr, ap, c=gain, cmap="coolwarm", s=45, edgecolor="k", linewidth=0.3, vmin=-0.5, vmax=0.5)
    fig.colorbar(sc, shrink=0.6, pad=0.1, label="AP gain of LSA over BM25")
    ax.set_xlabel("Query length (terms)"); ax.set_ylabel("# relevant documents"); ax.set_zlabel("BM25 AP")
    ax.set_title("Query difficulty in 3D"); ax.view_init(25, -60); save("adv_09_3d_query_difficulty.png")

# ---- 10. Pairwise significance heatmap
def fig10():
    n_ = len(names); D = np.zeros((n_, n_)); P = np.ones((n_, n_))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j: continue
            x, y = pq[a]["MAP"], pq[b]["MAP"]; D[i, j] = (x - y).mean()
            try: P[i, j] = wilcoxon(x, y).pvalue
            except ValueError: P[i, j] = 1
    fig, ax = plt.subplots(figsize=(7.5, 6.5)); m = np.abs(D).max() or 1
    im = ax.imshow(D, cmap="RdYlGn", vmin=-m, vmax=m)
    for i in range(n_):
        for j in range(n_):
            if i != j:
                star = "***" if P[i, j] < .001 else "**" if P[i, j] < .01 else "*" if P[i, j] < .05 else ""
                ax.text(j, i, f"{D[i,j]:+.3f}\n{star}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(n_)); ax.set_xticklabels(names, rotation=35); ax.set_yticks(range(n_)); ax.set_yticklabels(names)
    ax.set_title("Mean AP difference (row - column)\n* p<.05  ** p<.01  *** p<.001 (Wilcoxon)")
    plt.colorbar(im, fraction=0.045); save("adv_10_pairwise_significance.png")

for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]: safe(f)
print("\nDone. Figures are in ./results/adv_*.png")
