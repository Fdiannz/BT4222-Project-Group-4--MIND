"""Shared evaluation. Owner: P2.

Every model is scored with these functions so results are comparable.

Expected input: a candidate-level DataFrame with one row per displayed
candidate on FULL slates (no negative sampling), containing
    impression_id, news_id, label (0/1), and a score column.

Usage:
    from src.eval import evaluate, evaluate_by_segment, topk_diversity
    evaluate(df, "score")                         # overall
    evaluate_by_segment(df, "score", "is_cold")   # cold vs warm impressions
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------- per-impression relevance metrics (MIND official style) ----------

def _dcg(labels_in_rank_order, k):
    gains = labels_in_rank_order[:k]
    return float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))


def _impression_metrics(labels, scores):
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    ideal = np.sort(labels)[::-1]
    first_hit = np.flatnonzero(ranked)[0] + 1
    return {
        "auc": roc_auc_score(labels, scores),
        # MIND official MRR: mean reciprocal rank over ALL clicked items
        # (a perfect ranking with 2 clicks gives (1 + 1/2) / 2 = 0.75)
        "mrr": float(np.mean(1.0 / (np.flatnonzero(ranked) + 1))),
        "ndcg@5": _dcg(ranked, 5) / _dcg(ideal, 5),
        "ndcg@10": _dcg(ranked, 10) / _dcg(ideal, 10),
        "first_hit_rank": first_hit,
    }


def evaluate(df, score_col, label_col="label", group_col="impression_id", per_impression=False):
    """Mean AUC, MRR, nDCG@5, nDCG@10 over impressions.

    Impressions with no click or no non-click are skipped (AUC undefined)
    and counted in 'n_skipped'.
    """
    rows, skipped = [], 0
    for imp, g in df.groupby(group_col, sort=False):
        labels = g[label_col].to_numpy()
        if labels.min() == labels.max():
            skipped += 1
            continue
        m = _impression_metrics(labels, g[score_col].to_numpy(dtype=float))
        m[group_col] = imp
        rows.append(m)
    res = pd.DataFrame(rows)
    if per_impression:
        return res
    summary = res[["auc", "mrr", "ndcg@5", "ndcg@10"]].mean().round(4).to_dict()
    summary["n_impressions"] = len(res)
    summary["n_skipped"] = skipped
    return summary


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


def evaluate_by_segment(df, score_col, flag_col, label_col="label", group_col="impression_id"):
    """Relevance metrics overall and split into cold / warm / mixed impressions."""
    seg = label_impression_segment(df, flag_col, label_col, group_col)
    out = {"all": evaluate(df, score_col, label_col, group_col)}
    df = df.join(seg, on=group_col)
    for s in ["cold", "warm", "mixed"]:
        sub = df[df["segment"] == s]
        if len(sub):
            out[s] = evaluate(sub, score_col, label_col, group_col)
    return pd.DataFrame(out).T


# ---------- top-k list metrics (diversity, fair exposure) ----------

def topk(df, rank_col, k=10, group_col="impression_id", ascending=True):
    """Top-k rows per impression by rank_col (rank 1 = best when ascending)."""
    return (df.sort_values([group_col, rank_col], ascending=[True, ascending])
              .groupby(group_col, sort=False).head(k))


def topk_diversity(top, category_col="category", emb_lookup=None, news_col="news_id",
                   cold_col=None, group_col="impression_id"):
    """Diversity / exposure metrics for a top-k DataFrame (output of topk()).

    - unique_categories: mean number of distinct categories per list
    - intra_list_sim: mean pairwise cosine similarity (needs emb_lookup:
      dict news_id -> L2-normalised vector); lower = more diverse
    - cold_share: mean share of cold articles per list (needs cold_col)
    """
    res = {"unique_categories": top.groupby(group_col)[category_col].nunique().mean()}
    if emb_lookup is not None:
        sims = []
        for _, g in top.groupby(group_col, sort=False):
            vecs = np.stack([emb_lookup[n] for n in g[news_col]])
            if len(vecs) < 2:
                continue
            s = vecs @ vecs.T
            iu = np.triu_indices(len(vecs), k=1)
            sims.append(s[iu].mean())
        res["intra_list_sim"] = float(np.mean(sims))
    if cold_col is not None:
        res["cold_share"] = top.groupby(group_col)[cold_col].mean().mean()
    res["n_lists"] = top[group_col].nunique()
    return {k: (round(float(v), 4) if isinstance(v, float) else int(v)) for k, v in res.items()}
