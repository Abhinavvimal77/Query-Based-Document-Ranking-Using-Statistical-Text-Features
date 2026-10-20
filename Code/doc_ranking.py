"""
Project 23: Query-Based Document Ranking Using Statistical Text Features
Dataset : Cranfield collection (1400 docs, 225 queries, graded relevance judgements)
Models  : TF, TF-IDF (cosine), BM25, Query-Likelihood (Dirichlet), LSA, Hybrid (BM25+LSA)
Metrics : P@10, R@10, MAP, MRR, nDCG@10  (+ Wilcoxon significance, ablations, error analysis)

Install : pip install ir_datasets nltk scikit-learn scipy pandas matplotlib numpy
Run     : python doc_ranking.py            (real Cranfield data)
          python doc_ranking.py --toy      (tiny synthetic data, only to test the code)
"""
import re, sys, os, itertools, warnings
from functools import lru_cache
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore")
OUT = "results"; os.makedirs(OUT, exist_ok=True)
K = 10  # cut-off

# ---------------------------------------------------------------- 1. DATA
def load_cranfield():
    import ir_datasets
    ds = ir_datasets.load("cranfield")
    docs = {d.doc_id: (d.title + " " + d.text) for d in ds.docs_iter()}
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    qrels = {}
    for r in ds.qrels_iter():
        if r.relevance > 0:                      # keep judged-relevant docs, keep grade for nDCG
            qrels.setdefault(r.query_id, {})[r.doc_id] = r.relevance
    return docs, queries, qrels

def load_toy():
    rng = np.random.default_rng(0)
    topics = {"t%d" % i: ["w%d_%d" % (i, j) for j in range(15)] for i in range(5)}
    common = ["the", "of", "and", "flow", "system"]
    docs, queries, qrels = {}, {}, {}
    for i, (t, ws) in enumerate(topics.items()):
        qid = "q%d" % i
        queries[qid] = " ".join(rng.choice(ws, 4)) + " the of"
        qrels[qid] = {}
        for k in range(20):
            did = "d%d_%d" % (i, k)
            docs[did] = " ".join(list(rng.choice(ws, 12)) + list(rng.choice(common, 10)))
            qrels[qid][did] = int(rng.integers(1, 4))
    return docs, queries, qrels

# ---------------------------------------------------------------- 2. PREPROCESSING
_stem = PorterStemmer()
@lru_cache(maxsize=None)
def stem(w): return _stem.stem(w)

def preprocess(text, lower=True, stop=True, stemming=True):
    text = text.lower() if lower else text
    toks = re.findall(r"[a-zA-Z0-9]+", text)
    if stop:     toks = [t for t in toks if t.lower() not in ENGLISH_STOP_WORDS]
    if stemming: toks = [stem(t) for t in toks]
    return toks

# ---------------------------------------------------------------- 3. FEATURES
class Index:
    """Term-document statistics shared by all models."""
    def __init__(self, docs, queries, **pp):
        self.doc_ids = list(docs); self.q_ids = list(queries)
        dt = [preprocess(docs[d], **pp) for d in self.doc_ids]
        qt = [preprocess(queries[q], **pp) for q in self.q_ids]
        self.cv = CountVectorizer(analyzer=lambda x: x)
        self.X = self.cv.fit_transform(dt).tocsr().astype(float)          # docs x terms (tf)
        self.Q = self.cv.transform(qt).tocsr().astype(float)              # queries x terms (qtf)
        self.N = self.X.shape[0]
        self.dl = np.asarray(self.X.sum(1)).ravel()
        self.avgdl = self.dl.mean()
        self.df = np.asarray((self.X > 0).sum(0)).ravel()
        self.cf = np.asarray(self.X.sum(0)).ravel()
        self.pc = self.cf / self.cf.sum()                                  # collection LM
        self.vocab_size = self.X.shape[1]

def _scores(W, Qm):  # (queries x terms) @ (terms x docs)
    return np.asarray((Qm @ W.T).todense())

# ---------------------------------------------------------------- 4. MODELS (each returns queries x docs score matrix)
def score_tf(ix):
    return _scores(ix.X, (ix.Q > 0).astype(float))

def score_tfidf(ix, sublinear=True):
    tf = ix.X.copy()
    if sublinear: tf.data = 1 + np.log(tf.data)
    idf = np.log((ix.N + 1) / (ix.df + 1)) + 1
    W = normalize(tf @ sp.diags(idf))                                     # L2-normalised doc vectors
    Qv = normalize(ix.Q @ sp.diags(idf))
    return _scores(W, Qv)

def score_bm25(ix, k1=1.2, b=0.75):
    X = ix.X.tocoo()
    idf = np.log(1 + (ix.N - ix.df + 0.5) / (ix.df + 0.5))
    norm = k1 * (1 - b + b * ix.dl[X.row] / ix.avgdl)
    data = idf[X.col] * X.data * (k1 + 1) / (X.data + norm)
    W = sp.csr_matrix((data, (X.row, X.col)), shape=ix.X.shape)
    return _scores(W, (ix.Q > 0).astype(float))

def score_qlm(ix, mu=300):
    X = ix.X.tocoo()
    data = np.log(1 + X.data / (mu * ix.pc[X.col]))
    W = sp.csr_matrix((data, (X.row, X.col)), shape=ix.X.shape)
    s = _scores(W, ix.Q)
    qlen = np.asarray(ix.Q.sum(1)).ravel()[:, None]
    return s + qlen * np.log(mu / (ix.dl[None, :] + mu))                  # doc-length prior term

def score_lsa(ix, dims=200):
    idf = np.log((ix.N + 1) / (ix.df + 1)) + 1
    tf = ix.X.copy(); tf.data = 1 + np.log(tf.data)
    A = normalize(tf @ sp.diags(idf))
    svd = TruncatedSVD(n_components=min(dims, ix.vocab_size - 1), random_state=42)
    D = normalize(svd.fit_transform(A))
    Qr = normalize(svd.transform(normalize(ix.Q @ sp.diags(idf))))
    return Qr @ D.T

def minmax(S):
    mn, mx = S.min(1, keepdims=True), S.max(1, keepdims=True)
    return (S - mn) / (mx - mn + 1e-12)

def score_hybrid(ix, alpha=0.7, **kw):
    return alpha * minmax(score_bm25(ix)) + (1 - alpha) * minmax(score_lsa(ix, **kw))

# ---------------------------------------------------------------- 5. EVALUATION
def per_query_metrics(S, ix, qrels, k=K):
    rows = []
    for i, q in enumerate(ix.q_ids):
        if q not in qrels: continue
        rel = qrels[q]
        order = np.argsort(-S[i], kind="stable")
        ranked = [ix.doc_ids[j] for j in order]
        gains = np.array([rel.get(d, 0) for d in ranked])
        binary = (gains > 0).astype(float)
        R = len(rel)
        hits = binary.cumsum()
        prec_at = hits / (np.arange(len(ranked)) + 1)
        ap = (prec_at * binary).sum() / R
        first = np.where(binary > 0)[0]
        rr = 1.0 / (first[0] + 1) if len(first) else 0.0
        dcg = (gains[:k] / np.log2(np.arange(2, k + 2))).sum()
        ideal = np.sort(list(rel.values()))[::-1][:k]
        idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
        rows.append(dict(qid=q, **{"P@10": binary[:k].mean(), "R@10": binary[:k].sum() / R,
                                   "MAP": ap, "MRR": rr, "nDCG@10": dcg / idcg}))
    return pd.DataFrame(rows).set_index("qid")

def pr_curve(S, ix, qrels):
    grid = np.linspace(0, 1, 11); curves = []
    for i, q in enumerate(ix.q_ids):
        if q not in qrels: continue
        order = np.argsort(-S[i], kind="stable")
        b = np.array([ix.doc_ids[j] in qrels[q] for j in order], float)
        prec = b.cumsum() / (np.arange(len(b)) + 1); rec = b.cumsum() / len(qrels[q])
        curves.append([prec[rec >= r].max() if (rec >= r).any() else 0 for r in grid])  # interpolated
    return grid, np.mean(curves, 0)


# ---------------------------------------------------------------- 7. TUNING WITH TRAIN/TEST QUERY SPLIT
def tuned_experiment(ix, qrels, seed=42):
    """Tune all hyper-parameters on 50% of queries (train), report on the other 50% (test)."""
    rng = np.random.default_rng(seed)
    qs = sorted(qrels); qs = [qs[i] for i in rng.permutation(len(qs))]
    half = len(qs) // 2
    qtr = {q: qrels[q] for q in qs[:half]}; qte = {q: qrels[q] for q in qs[half:]}
    print(f"\n=== TUNED EXPERIMENT: {len(qtr)} train queries / {len(qte)} test queries ===")
    mp = lambda S, qr: per_query_metrics(S, ix, qr)["MAP"].mean()

    # BM25 (k1, b)
    k1s, bs = [0.5, 0.9, 1.2, 1.5, 2.0, 3.0, 4.0], [0.25, 0.5, 0.75, 1.0]
    g = {(k, b): mp(score_bm25(ix, k, b), qtr) for k in k1s for b in bs}
    k1_t, b_t = max(g, key=g.get)
    # QL mu
    mus = [50, 100, 200, 300, 500, 1000]
    mu_t = max(mus, key=lambda m: mp(score_qlm(ix, m), qtr))
    # LSA dimensions
    dims = [d for d in [25, 50, 100, 150, 200, 300, 400] if d < ix.vocab_size - 1] or [max(2, ix.vocab_size // 2)]
    lsa_cache = {d: score_lsa(ix, dims=d) for d in dims}
    lsa_curve = pd.Series({d: mp(lsa_cache[d], qtr) for d in dims}); d_t = lsa_curve.idxmax()
    # Hybrid alpha (using tuned BM25 + tuned LSA)
    bm_t = score_bm25(ix, k1_t, b_t)
    alphas = np.round(np.linspace(0, 1, 11), 1)
    a_curve = pd.Series({a: mp(a * minmax(bm_t) + (1 - a) * minmax(lsa_cache[d_t]), qtr) for a in alphas})
    a_t = a_curve.idxmax()
    best = dict(bm25_k1=k1_t, bm25_b=b_t, qlm_mu=mu_t, lsa_dims=int(d_t), hybrid_alpha=float(a_t))
    print("Best parameters (chosen on TRAIN):", best)
    pd.Series(best).to_csv(f"{OUT}/tuned_params.csv")
    lsa_curve.round(4).to_csv(f"{OUT}/lsa_dims_train.csv"); a_curve.round(4).to_csv(f"{OUT}/hybrid_alpha_train.csv")

    # Test-set comparison: default vs tuned
    M = {"TF-IDF cosine": score_tfidf(ix),
         "BM25 (default)": score_bm25(ix), "BM25 (tuned)": bm_t,
         "QL (default)": score_qlm(ix), "QL (tuned)": score_qlm(ix, mu_t),
         "LSA (default 200d)": score_lsa(ix, dims=min(200, ix.vocab_size - 1)), "LSA (tuned)": lsa_cache[d_t],
         "Hybrid (default a=0.7)": score_hybrid(ix, dims=min(200, ix.vocab_size - 1)),
         "Hybrid (tuned)": a_t * minmax(bm_t) + (1 - a_t) * minmax(lsa_cache[d_t])}
    pqt = {n: per_query_metrics(S, ix, qte) for n, S in M.items()}
    tab = pd.DataFrame({n: d.mean() for n, d in pqt.items()}).T.round(4)
    print("\n=== TEST-SET results (held-out queries) ===\n", tab); tab.to_csv(f"{OUT}/test_results_tuned.csv")

    rows = []
    for n in M:
        if n == "TF-IDF cosine": continue
        a, b_ = pqt[n]["MAP"], pqt["TF-IDF cosine"]["MAP"]
        try: p = wilcoxon(a, b_).pvalue
        except ValueError: p = np.nan
        rows.append(dict(model=n, vs="TF-IDF cosine", mean_AP_diff=round((a - b_).mean(), 4), p_value=round(p, 4)))
    for x, y in [("Hybrid (tuned)", "LSA (tuned)"), ("Hybrid (tuned)", "BM25 (tuned)")]:
        try: p = wilcoxon(pqt[x]["MAP"], pqt[y]["MAP"]).pvalue
        except ValueError: p = np.nan
        rows.append(dict(model=x, vs=y, mean_AP_diff=round((pqt[x]["MAP"] - pqt[y]["MAP"]).mean(), 4), p_value=round(p, 4)))
    sg = pd.DataFrame(rows); print("\n=== Wilcoxon on TEST queries (AP) ===\n", sg); sg.to_csv(f"{OUT}/significance_test.csv", index=False)

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    lsa_curve.plot(marker="o", ax=ax[0]); ax[0].set_xlabel("LSA dimensions"); ax[0].set_ylabel("MAP (train)"); ax[0].set_title("LSA dimensionality")
    a_curve.plot(marker="o", ax=ax[1]); ax[1].set_xlabel("alpha (weight of BM25)"); ax[1].set_ylabel("MAP (train)"); ax[1].set_title("Hybrid fusion weight")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_lsa_alpha_tuning.png", dpi=150); plt.close()
    ax = tab[["P@10", "R@10", "MAP", "MRR", "nDCG@10"]].plot.bar(figsize=(11, 5), rot=25)
    ax.set_title("Held-out test queries: default vs tuned"); plt.tight_layout()
    plt.savefig(f"{OUT}/fig_test_tuned.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 6. MAIN EXPERIMENTS
def main(toy=False):
    docs, queries, qrels = load_toy() if toy else load_cranfield()
    print(f"Docs={len(docs)} Queries={len(queries)} Judged queries={len(qrels)} "
          f"Avg rel/query={np.mean([len(v) for v in qrels.values()]):.1f}")

    ix = Index(docs, queries)
    print(f"Vocab={ix.vocab_size} avgdl={ix.avgdl:.1f}")

    # ---- Comparative analysis
    models = {"TF (overlap)": score_tf(ix), "TF-IDF cosine": score_tfidf(ix),
              "BM25": score_bm25(ix), "QL-Dirichlet": score_qlm(ix),
              "LSA (200d)": score_lsa(ix, dims=min(200, ix.vocab_size - 1)),
              "Hybrid BM25+LSA": score_hybrid(ix, dims=min(200, ix.vocab_size - 1))}
    pq = {n: per_query_metrics(S, ix, qrels) for n, S in models.items()}
    summary = pd.DataFrame({n: d.mean() for n, d in pq.items()}).T.round(4)
    print("\n=== Model comparison ===\n", summary)
    summary.to_csv(f"{OUT}/model_comparison.csv")

    # ---- Significance vs BM25 and vs TF-IDF (Wilcoxon on per-query AP)
    sig = []
    base = "TF-IDF cosine"
    for n in models:
        if n == base: continue
        a, b_ = pq[n]["MAP"], pq[base]["MAP"]
        try: p = wilcoxon(a, b_).pvalue
        except ValueError: p = np.nan
        sig.append(dict(model=n, baseline=base, mean_AP_diff=round((a - b_).mean(), 4), p_value=round(p, 4)))
    sig = pd.DataFrame(sig); print("\n=== Wilcoxon (AP) ===\n", sig); sig.to_csv(f"{OUT}/significance.csv", index=False)

    # ---- Ablation: preprocessing (BM25)
    abl = {}
    for name, pp in {"raw (no stop, no stem)": dict(stop=False, stemming=False),
                     "stopword removal": dict(stop=True, stemming=False),
                     "stemming only": dict(stop=False, stemming=True),
                     "stop + stem": dict(stop=True, stemming=True)}.items():
        i2 = Index(docs, queries, **pp)
        abl[name] = per_query_metrics(score_bm25(i2), i2, qrels).mean()
    abl = pd.DataFrame(abl).T.round(4); print("\n=== Preprocessing ablation (BM25) ===\n", abl)
    abl.to_csv(f"{OUT}/ablation_preprocessing.csv")

    # ---- Parameter sensitivity: BM25 k1 x b (MAP)
    k1s, bs = [0.5, 0.9, 1.2, 1.5, 2.0, 3.0], [0.0, 0.25, 0.5, 0.75, 1.0]
    grid = pd.DataFrame([[per_query_metrics(score_bm25(ix, k1, b), ix, qrels)["MAP"].mean()
                          for b in bs] for k1 in k1s], index=k1s, columns=bs).round(4)
    print("\n=== BM25 MAP grid (rows k1, cols b) ===\n", grid); grid.to_csv(f"{OUT}/bm25_grid.csv")
    best_k1, best_b = np.unravel_index(grid.values.argmax(), grid.shape)
    print("Best:", k1s[best_k1], bs[best_b])
    mus = [50, 100, 200, 300, 500, 1000, 2000]
    mu_df = pd.Series({m: per_query_metrics(score_qlm(ix, m), ix, qrels)["MAP"].mean() for m in mus}).round(4)
    mu_df.to_csv(f"{OUT}/qlm_mu.csv"); print("\n=== QL mu sweep (MAP) ===\n", mu_df)

    # ---- Plots
    ax = summary.plot.bar(figsize=(10, 5), rot=20); ax.set_title("Model comparison"); ax.set_ylabel("score")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_model_comparison.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4.5)); plt.imshow(grid.values, cmap="viridis", aspect="auto")
    plt.xticks(range(len(bs)), bs); plt.yticks(range(len(k1s)), k1s); plt.colorbar(label="MAP")
    plt.xlabel("b"); plt.ylabel("k1"); plt.title("BM25 parameter sensitivity")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_bm25_grid.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4.5))
    for n in ["TF-IDF cosine", "BM25", "QL-Dirichlet", "LSA (200d)", "Hybrid BM25+LSA"]:
        g, p = pr_curve(models[n], ix, qrels); plt.plot(g, p, marker="o", ms=3, label=n)
    plt.xlabel("Recall"); plt.ylabel("Interpolated precision"); plt.legend(); plt.title("11-point P-R curve")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_pr_curve.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4)); pq["BM25"]["MAP"].plot.hist(bins=20, edgecolor="k")
    plt.xlabel("Average precision per query"); plt.title("BM25: per-query AP distribution")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_ap_hist.png", dpi=150); plt.close()

    # ---- Error analysis (BM25)
    S = models["BM25"]; worst = pq["BM25"].sort_values("MAP").head(5).index
    lines = []
    for q in worst:
        i = ix.q_ids.index(q); order = np.argsort(-S[i])
        rank_of = {ix.doc_ids[j]: r + 1 for r, j in enumerate(order)}
        qtoks = preprocess(queries[q]); in_vocab = [t for t in qtoks if t in ix.cv.vocabulary_]
        rel_ranks = sorted(rank_of[d] for d in qrels[q])
        covered = np.mean([any(t in preprocess(docs[d]) for t in qtoks) for d in qrels[q]])
        lines.append(f"QUERY {q}: {queries[q]}\n  AP={pq['BM25'].loc[q,'MAP']:.3f} | #rel={len(qrels[q])} "
                     f"| best relevant ranks={rel_ranks[:5]}\n  query terms={qtoks} | rel. docs sharing >=1 term={covered:.0%}\n"
                     f"  top-1 retrieved (non-relevant?): {ix.doc_ids[order[0]]} -> {docs[ix.doc_ids[order[0]]][:150]}...\n")
    open(f"{OUT}/error_analysis.txt", "w").write("\n".join(lines)); print("\n=== Error analysis ===\n" + "\n".join(lines))
    corr = pd.DataFrame({"qlen": [len(preprocess(queries[q])) for q in pq["BM25"].index],
                         "n_rel": [len(qrels[q]) for q in pq["BM25"].index],
                         "AP": pq["BM25"]["MAP"].values}).corr(method="spearman")["AP"]
    print("Spearman(AP vs query length / #relevant):\n", corr.round(3)); corr.to_csv(f"{OUT}/error_correlation.csv")

    # ---- Tuned experiment with train/test split
    tuned_experiment(ix, qrels)

    # ---- Demo: ad-hoc query
    def search(text, top=5):
        qv = ix.cv.transform([preprocess(text)]).astype(float)
        ix2 = ix; ix2.Q = qv; s = score_bm25(ix2)[0]
        ix.Q = ix.cv.transform([preprocess(queries[q]) for q in ix.q_ids]).astype(float)
        return [(ix.doc_ids[j], round(s[j], 3), docs[ix.doc_ids[j]][:80]) for j in np.argsort(-s)[:top]]
    sample = "boundary layer transition at supersonic speeds" if not toy else "w0_1 w0_2"
    print("\n=== Demo search:", sample, "===")
    for r in search(sample): print(r)
    print(f"\nAll tables/figures saved in ./{OUT}/")

if __name__ == "__main__":
    main(toy="--toy" in sys.argv)
