"""Shared metrics (utils/metrics.py). Owner: P2.

Every model is scored with these functions so results are comparable.

Expected input: a candidate-level DataFrame with one row per displayed
candidate on FULL slates (no negative sampling), containing
    impression_id, news_id, label (0/1), and a score column
(plus, optionally, is_cold and category for the segment / diversity metrics).

Usage:
    from utils.metrics import evaluate, evaluate_by_segment, topk, topk_diversity
    evaluate(df, "score")                         # overall
    evaluate_by_segment(df, "score", "is_cold")   # cold / warm / mixed impressions

    df = add_rank(df, "score")                    # score -> rank (1 = best)
    top = topk(df, "rank", k=10)                  # top-10 list per impression
    topk_diversity(top, "category", emb_lookup, cold_col="is_cold")

    sweep_table({"lambda=1.0": df_a, "lambda=0.8": df_b}, emb_lookup=emb)  # one row per model

Definitions follow the official MIND evaluation (per-impression metrics, then averaged):
    auc      ROC-AUC inside the impression, ties count 0.5
    mrr      mean of 1/rank over ALL clicked items (perfect ranking, 2 clicks -> 0.75)
    ndcg@k   binary gain, discount 1/log2(rank+1), normalised by the ideal list
Impressions with no click or no non-click are skipped (AUC undefined) and counted
in 'n_skipped'.
Ties in score are broken randomly but reproducibly (SEED), never by row order, so a
model that outputs many equal scores (e.g. popularity) is neither helped nor hurt by
how the file happens to be sorted.
"""
import numpy as np
import pandas as pd

try:  # shared settings live in config.py
    from config import SEED
except Exception:  # pragma: no cover
    SEED = 42

KS = (5, 10)


# ---------------------------------------------------------------- ranking helper
def _rank_within_group(iid, score, seed):
    """Sort rows by (impression, score desc, random tie-break) and rank inside each impression."""
    tie = np.random.default_rng(seed).random(len(score))
    order = np.lexsort((tie, -score, iid))
    iid_s = iid[order]
    new_grp = np.r_[True, iid_s[1:] != iid_s[:-1]]
    grp = np.cumsum(new_grp) - 1
    first = np.flatnonzero(new_grp)
    rank = np.arange(len(iid_s)) - first[grp] + 1
    return order, grp, first, rank


def add_rank(df, score_col, rank_col="rank", group_col="impression_id", seed=SEED):
    """Add a 1-based rank (1 = highest score) within each impression. Row order is unchanged."""
    iid = pd.factorize(df[group_col])[0]
    order, _, _, rank_s = _rank_within_group(iid, df[score_col].to_numpy(dtype=float), seed)
    rank = np.empty(len(df), dtype=np.int64)
    rank[order] = rank_s
    out = df.copy()
    out[rank_col] = rank
    return out


# ------------------------------------------------- per-impression relevance metrics
def _per_impression(df, score_col, label_col, group_col, seed=SEED):
    """One row per impression: auc, mrr, ndcg@5, ndcg@10, first_hit_rank (vectorised)."""
    iid = pd.factorize(df[group_col])[0]
    score = df[score_col].to_numpy(dtype=float)
    order, grp, first, rank = _rank_within_group(iid, score, seed)
    label_s = df[label_col].to_numpy(dtype=int)[order]
    score_s = score[order]

    n_cand = np.bincount(grp)
    n_pos = np.bincount(grp, weights=label_s)
    n_neg = n_cand - n_pos
    res = pd.DataFrame({group_col: df[group_col].to_numpy()[order][first]})

    with np.errstate(divide="ignore", invalid="ignore"):
        res["mrr"] = np.bincount(grp, weights=label_s / rank) / n_pos
        for k in KS:
            dcg = np.bincount(grp, weights=label_s * (rank <= k) / np.log2(rank + 1))
            ideal = np.r_[0.0, np.cumsum(1.0 / np.log2(np.arange(1, k + 1) + 1))]
            res[f"ndcg@{k}"] = dcg / ideal[np.minimum(n_pos, k).astype(int)]
        # AUC from rank sums (ascending ranks, ties averaged -> a tie counts 0.5)
        asc = pd.Series(score_s).groupby(grp).rank(method="average").to_numpy()
        auc = (np.bincount(grp, weights=label_s * asc) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    res["auc"] = auc

    hit = label_s == 1
    ug, idx = np.unique(grp[hit], return_index=True)   # first click = smallest rank in its impression
    first_hit = np.full(len(n_cand), np.nan)
    first_hit[ug] = rank[hit][idx]
    res["first_hit_rank"] = first_hit
    res["valid"] = (n_pos > 0) & (n_neg > 0)
    return res


def _summary(res):
    """Mean metrics over the valid impressions of `res` (output of _per_impression)."""
    ok = res[res["valid"]]
    out = ok[["auc", "mrr", "ndcg@5", "ndcg@10"]].mean().round(4).to_dict()
    out["n_impressions"] = int(len(ok))
    out["n_skipped"] = int(len(res) - len(ok))
    return out


def evaluate(df, score_col, label_col="label", group_col="impression_id", per_impression=False,
             seed=SEED):
    """Mean AUC, MRR, nDCG@5, nDCG@10 over impressions.

    Impressions with no click or no non-click are skipped (AUC undefined)
    and counted in 'n_skipped'.
    """
    res = _per_impression(df, score_col, label_col, group_col, seed)
    if per_impression:
        ok = res[res["valid"]]
        return ok[["auc", "mrr", "ndcg@5", "ndcg@10", "first_hit_rank", group_col]].reset_index(drop=True)
    return _summary(res)


def label_impression_segment(df, flag_col, label_col="label", group_col="impression_id"):
    """Label each impression by its CLICKED articles' flag (e.g. is_cold).

    'cold'  = all clicked articles have flag 1
    'warm'  = all clicked articles have flag 0
    'mixed' = some of each
    Returns a Series indexed by impression_id.
    """
    clicked = df[df[label_col] == 1].groupby(group_col)[flag_col].agg(["min", "max"])
    seg = np.where(clicked["min"] == 1, "cold", np.where(clicked["max"] == 0, "warm", "mixed"))
    return pd.Series(seg, index=clicked.index, name="segment")


def evaluate_by_segment(df, score_col, flag_col, label_col="label", group_col="impression_id",
                        seed=SEED):
    """Relevance metrics overall and split into cold / warm / mixed impressions."""
    res = _per_impression(df, score_col, label_col, group_col, seed)
    res["segment"] = res[group_col].map(label_impression_segment(df, flag_col, label_col, group_col))
    rows = {"all": _summary(res)}
    for s in ["cold", "warm", "mixed"]:
        sub = res[res["segment"] == s]
        if len(sub):
            rows[s] = _summary(sub)
    out = pd.DataFrame(rows).T
    out[["n_impressions", "n_skipped"]] = out[["n_impressions", "n_skipped"]].astype(int)
    return out


# --------------------------------------------- top-k list metrics (diversity, exposure)
def topk(df, rank_col, k=10, group_col="impression_id", ascending=True):
    """Top-k rows per impression by rank_col (rank 1 = best when ascending).

    No rank column yet?  df = add_rank(df, "score")  then  topk(df, "rank").
    """
    return (df.sort_values([group_col, rank_col], ascending=[True, ascending])
              .groupby(group_col, sort=False).head(k))


def topk_diversity(top, category_col="category", emb_lookup=None, news_col="news_id",
                   cold_col=None, group_col="impression_id", k=None):
    """Diversity / exposure metrics for a top-k DataFrame (output of topk()).

    - unique_categories: mean number of distinct categories per list
    - intra_list_sim: mean pairwise cosine similarity (needs emb_lookup:
      dict news_id -> vector); lower = more diverse
    - cold_share: mean share of cold articles per list (needs cold_col)

    Only lists with exactly k items are used (k defaults to the longest list). A slate with
    10 or fewer candidates has nothing to re-rank (interim report, Section 1.2), so its
    short list is left out and counted in 'n_short_lists_excluded'.
    """
    size = top.groupby(group_col, sort=False)[news_col].transform("size")
    k = int(size.max()) if k is None else k
    full = top[size == k].sort_values(group_col, kind="stable")
    res = {"unique_categories": full.groupby(group_col)[category_col].nunique().mean()}
    if emb_lookup is not None and k > 1:
        ids = pd.Index(list(emb_lookup.keys()))
        E = np.stack(list(emb_lookup.values())).astype(np.float32)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        idx = ids.get_indexer(full[news_col])
        if (idx < 0).any():
            raise ValueError(f"{int((idx < 0).sum())} top-k articles are missing from emb_lookup")
        idx = idx.reshape(-1, k)
        sim = np.empty(len(idx))
        for s in range(0, len(idx), 20000):          # chunked to keep memory small
            v = E[idx[s:s + 20000]]
            sim[s:s + 20000] = (np.einsum("nkd,njd->nkj", v, v).sum((1, 2)) - k) / (k * (k - 1))
        res["intra_list_sim"] = float(sim.mean())
    if cold_col is not None:
        res["cold_share"] = full.groupby(group_col)[cold_col].mean().mean()
    res["n_lists"] = full[group_col].nunique()
    res["n_short_lists_excluded"] = top[group_col].nunique() - res["n_lists"]
    return {key: (round(float(v), 4) if isinstance(v, float) else int(v)) for key, v in res.items()}


# --------------------------------------------------- score files and the lambda sweep
def attach_scores(labels, scores):
    """Join a model's score file (impression_id, news_id, score) onto the labelled candidates.

    Raises if any candidate has no score or the score file has duplicates / NaN / extra rows.
    A silent gap would make a model look better than it is.
    """
    key = ["impression_id", "news_id"]
    if scores.duplicated(key).any():
        raise ValueError("score file has duplicate (impression_id, news_id) rows")
    if scores["score"].isna().any():
        raise ValueError("score file contains NaN scores")
    out = labels.merge(scores[key + ["score"]], on=key, how="left", validate="one_to_one")
    if out["score"].isna().any():
        raise ValueError(f"{int(out['score'].isna().sum())} candidates have no score - score ALL candidates")
    if len(scores) != len(labels):
        raise ValueError(f"score file has {len(scores)} rows, expected {len(labels)}")
    return out


def sweep_table(scored, score_col="score", flag_col="is_cold", category_col="category",
                emb_lookup=None, k=10):
    """One row per model / lambda: relevance by segment + diversity + cold share of the top-k.

    scored: {"popularity": df, "mmr lambda=0.8": df_2, ...}; each df is a labelled candidate
    table with the score column (re-rankers write their new order as the score column).
    """
    rows = {}
    for name, df in scored.items():
        rel = evaluate_by_segment(df, score_col, flag_col)
        top = topk(add_rank(df, score_col), "rank", k=k)
        div = topk_diversity(top, category_col, emb_lookup, cold_col=flag_col, k=k)
        row = {f"{seg}_{m}": rel.loc[seg, m] for seg in rel.index
               for m in ("auc", "mrr", "ndcg@5", "ndcg@10", "n_impressions")}
        row.update({f"top{k}_{key}": v for key, v in div.items()})
        rows[name] = row
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------ tests
def _selftest():
    """Hand-worked toy slates.  Run:  python -m utils.metrics   (or call from a notebook)."""
    def table(slates):
        rows = []
        for i, (lab, sc) in enumerate(slates):
            rows += [(f"imp{i}", f"n{i}_{j}", l, s) for j, (l, s) in enumerate(zip(lab, sc))]
        return pd.DataFrame(rows, columns=["impression_id", "news_id", "label", "score"])

    def one(lab, sc):
        return evaluate(table([(lab, sc)]), "score", per_impression=True).iloc[0]

    close = lambda a, b: abs(a - b) < 1e-9
    r = one([0, 1, 0, 0], [.9, .8, .7, .1])                      # clicked item ranked 2nd of 4
    assert close(r.mrr, .5) and close(r["ndcg@10"], 1 / np.log2(3)) and close(r.auc, 2 / 3)
    assert r.first_hit_rank == 2
    r = one([1, 0, 0], [3, 2, 1])                                # perfect
    assert close(r.auc, 1) and close(r.mrr, 1) and close(r["ndcg@5"], 1)
    r = one([1, 0, 0], [1, 2, 3])                                # worst
    assert close(r.auc, 0) and close(r.mrr, 1 / 3) and close(r["ndcg@10"], 1 / np.log2(4))
    r = one([1, 1, 0, 0], [.9, .1, .8, .7])                      # two clicks, ranks 1 and 4
    assert close(r.mrr, (1 + .25) / 2) and close(r.auc, .5)
    assert close(r["ndcg@10"], (1 + 1 / np.log2(5)) / (1 + 1 / np.log2(3)))
    r = one([0, 0, 0, 0, 0, 1], [6, 5, 4, 3, 2, 1])              # click ranked 6th: outside nDCG@5
    assert close(r["ndcg@5"], 0) and close(r.mrr, 1 / 6)
    assert close(one([1, 0], [1, 1]).auc, .5)                    # tie -> AUC 0.5
    s = evaluate(table([([1, 1], [2, 1]), ([1, 0], [2, 1])]), "score")   # all-clicked slate skipped
    assert s["n_skipped"] == 1 and s["n_impressions"] == 1 and s["mrr"] == 1

    # segments: imp0 clicked cold, imp1 clicked warm, imp2 clicked one of each
    df = table([([1, 0, 0], [3, 2, 1]), ([0, 1, 0], [3, 2, 1]), ([1, 1, 0], [3, 2, 1])])
    df["is_cold"] = [1, 0, 0, 0, 0, 1, 1, 0, 0]
    seg = evaluate_by_segment(df, "score", "is_cold")
    assert list(seg.index) == ["all", "cold", "warm", "mixed"]
    assert list(seg["n_impressions"]) == [3, 1, 1, 1] and seg.loc["warm", "mrr"] == .5

    # top-k lists: category variety, cold share, intra-list similarity (a, a, b -> pairs 1,0,0)
    d = table([([1, 0, 0, 0], [4, 3, 2, 1]), ([1, 0], [2, 1])])
    d["news_id"] = ["a1", "a2", "b", "x", "a1", "b"]
    d["category"] = ["sports", "sports", "news", "news", "sports", "news"]
    d["is_cold"] = [1, 0, 0, 0, 0, 0]
    top = topk(add_rank(d, "score"), "rank", k=3)
    emb = {"a1": [1., 0.], "a2": [1., 0.], "b": [0., 1.], "x": [1., 1.]}
    out = topk_diversity(top, "category", emb, cold_col="is_cold", k=3)
    assert out["unique_categories"] == 2 and out["n_lists"] == 1 and out["n_short_lists_excluded"] == 1
    assert abs(out["intra_list_sim"] - round(1 / 3, 4)) < 1e-9 and abs(out["cold_share"] - round(1 / 3, 4)) < 1e-9

    # ranks are reproducible and a tie never depends on row order
    t = table([([1, 0, 0, 0], [1, 1, 1, 1])])
    assert (add_rank(t, "score")["rank"].to_numpy() == add_rank(t, "score")["rank"].to_numpy()).all()

    # score-file checks
    lab = table([([1, 0], [0, 0])]).drop(columns="score")
    sc = lab[["impression_id", "news_id"]].assign(score=[.2, .1])
    assert len(attach_scores(lab, sc)) == 2
    for bad in (sc.iloc[:1], pd.concat([sc, sc.iloc[:1]])):
        try:
            attach_scores(lab, bad)
            raise AssertionError("should have raised")
        except ValueError:
            pass
    print("metrics self-test passed")


if __name__ == "__main__":
    _selftest()
